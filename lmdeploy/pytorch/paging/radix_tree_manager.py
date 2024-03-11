# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from ..block import LogicalTokenBlocks
from ..config import CacheConfig
from ..messages import SchedulerSequence
from .block_manager import BlockManager


def _div_up(a, b):
    """div up."""
    return (a + b - 1) // b


@dataclass
class TreeNode:
    """tree node."""
    node_id: int
    manager: 'RadixTreeManager'
    token_ids: np.ndarray
    blocks: np.ndarray
    parent: 'TreeNode' = None
    children: Dict[int, 'TreeNode'] = field(init=False, default_factory=dict)
    device: str = 'gpu'
    last_visit_time: int = 0
    sequence: SchedulerSequence = None

    def __setattr__(self, __name: str, __value: Any) -> None:
        """set attr."""

        def __clear_old_parent():
            old_parent = self.parent
            if old_parent is not None:
                old_parent.children.pop(self.node_id, None)

        def __update_new_parent():
            parent: TreeNode = __value
            if parent is not None:
                parent.children[self.node_id] = self

        if __name == 'parent':
            __clear_old_parent()
            __update_new_parent()
        super().__setattr__(__name, __value)

    @property
    def num_blocks(self):
        return len(self.blocks)

    @property
    def num_token_ids(self):
        return len(self.token_ids)

    @property
    def num_cum_blocks(self):
        """num cum blocks."""
        num_parent_blocks = (0 if self.parent is None else
                             self.parent.num_cum_blocks)
        return num_parent_blocks + self.num_blocks

    @property
    def num_cum_ids(self):
        """num cum ids."""
        num_parent_ids = (0
                          if self.parent is None else self.parent.num_cum_ids)
        return num_parent_ids + self.num_token_ids

    def update_visit_time(self, time: int, window_size: int = -1):
        """update visit time."""
        self.last_visit_time = time
        if self.parent is not None:
            if window_size < 0:
                self.parent.update_visit_time(time)
                return
            next_window_size = window_size - self.num_token_ids
            if next_window_size > 0:
                self.parent.update_visit_time(time, next_window_size)

    def __str__(self):
        children_str = [str(child) for child in self.children]
        children_str = ', '.join(children_str)
        parent = 'None' if self.parent is None else f'<{self.parent.node_id}>'
        return f"""TreeNode(
node_id={self.node_id},
token_ids={self.token_ids},
blocks={self.blocks},
parent={parent},
children=[{children_str}])"""

    def __repr__(self):
        return str(self)


def _np_empty(dtype=np.int64):
    return np.empty((0, ), dtype=dtype)


class RadixTreeManager:
    """Radix tree manager."""

    def __init__(self, cache_config: CacheConfig,
                 block_manager: BlockManager) -> None:

        def __build_root():
            """build root."""
            node = TreeNode(0,
                            manager=self,
                            token_ids=_np_empty(),
                            blocks=_np_empty(),
                            parent=None)
            return node

        def __build_max_node_blocks():
            """build max_node_blocks."""
            window_size = self.cache_config.window_size
            block_size = self.cache_config.block_size
            if window_size <= 0:
                window_size = block_size * 4096
            return _div_up(window_size, block_size)

        self.cache_config = cache_config
        self._block_manager = block_manager
        self._block_size = self.cache_config.block_size

        self._root = __build_root()
        self.max_node_blocks = __build_max_node_blocks()
        self.max_node_tokens = (self.max_node_blocks * self.block_size)

        self.nodes: Dict[int, TreeNode] = {self.root.node_id: self.root}
        self.seq_node_map: Dict[int, TreeNode] = dict()

        self._max_node_id = self.root.node_id + 1

    @property
    def step_time(self):
        return self._root.last_visit_time

    @property
    def block_size(self):
        return self._block_size

    @property
    def root(self):
        return self._root

    def _next_new_node_id(self):
        """next new node id."""
        ret = self._max_node_id
        self._max_node_id += 1
        return ret

    def update_map(self, seq: SchedulerSequence, node: TreeNode):
        """update map."""
        self.seq_node_map[seq.seq_id] = node
        node.sequence = seq

    def pop_map(self, *, seq: SchedulerSequence = None, node: TreeNode = None):
        """pop map."""

        def __pop_seq():
            """pop seq."""
            node = self.seq_node_map.pop(seq.seq_id, None)
            if node is not None:
                node.sequence = None

        def __pop_node():
            """pop node."""
            seq = node.sequence
            self.seq_node_map.pop(seq.seq_id, None)
            node.sequence = None

        if seq is not None:
            assert node is None
            __pop_seq()
        else:
            assert node is not None
            __pop_node()

    def num_appendable_ids(self, node: TreeNode):
        """last block appendable ids."""
        return node.num_blocks * self.block_size - node.num_token_ids

    def num_required_blocks(self, num_tokens: int, node: TreeNode = None):
        """num required blocks."""
        if node is not None:
            num_last_remain = self.num_appendable_ids(node)
            num_tokens = max(0, num_tokens - num_last_remain)
        return _div_up(num_tokens, self.cache_config.block_size)

    def new_node(self,
                 token_ids: np.ndarray,
                 blocks: np.ndarray,
                 parent: TreeNode = None,
                 children: Dict[int, TreeNode] = None,
                 device: str = 'gpu',
                 last_visit_time: int = None):
        """new node."""
        node_id = self._next_new_node_id()
        children = children or dict()
        parent = parent or self.root
        if last_visit_time is None:
            last_visit_time = self.step_time

        node = TreeNode(node_id=node_id,
                        manager=self,
                        token_ids=token_ids,
                        blocks=blocks,
                        parent=parent,
                        device=device,
                        last_visit_time=last_visit_time)
        for child in children.values():
            child.parent = node
        self.nodes[node_id] = node

        return node

    def add_node(self,
                 parent: TreeNode = None,
                 token_ids: np.ndarray = None,
                 blocks: np.ndarray = None,
                 device: str = 'gpu'):
        """add node.

        Add node should not allocate blocks, block allocation only happened
        when add or update sequence.
        """

        if parent is None:
            parent = self.root
        if token_ids is None:
            token_ids = _np_empty()
        if blocks is None:
            blocks = _np_empty()

        assert self.num_appendable_ids(parent) == 0

        num_tokens = len(token_ids)
        num_blocks = len(blocks)
        block_size = self.cache_config.block_size
        max_node_blocks = self.max_node_blocks
        max_node_tokens = self.max_node_tokens

        assert num_blocks * block_size >= num_tokens

        if num_blocks == 0:
            return self.new_node(token_ids=token_ids,
                                 blocks=blocks,
                                 parent=parent,
                                 device=device)

        for bidx in range(0, num_blocks, max_node_blocks):
            tidx = bidx * block_size
            tmp_blocks = blocks[bidx:bidx + max_node_blocks]
            tmp_token_ids = token_ids[tidx:tidx + max_node_tokens]
            node = self.new_node(token_ids=tmp_token_ids,
                                 blocks=tmp_blocks,
                                 parent=parent,
                                 device=device)
            parent = node

        return node

    def remove_node(self, node: TreeNode):
        """remove node."""

        def __update_leaf():
            """update leaf seq."""
            seq = node.sequence
            num_blocks = node.num_blocks
            num_ids = node.num_token_ids
            seq_blocks = seq.logical_blocks
            new_seq_blocks = LogicalTokenBlocks(seq_blocks[:-num_blocks])
            seq.logical_blocks = new_seq_blocks
            new_step = seq.history_len - num_ids
            seq.set_step(new_step)
            self.pop_map(node=node)
            new_node = self.add_node(node.parent)
            self.update_map(seq, new_node)

        def __update_empty_child(node: TreeNode, remove_len: int):
            """update empty child."""
            seq = node.sequence
            assert seq is not None
            seq_blocks = seq.logical_blocks
            new_seq_blocks = LogicalTokenBlocks(seq_blocks[:-remove_len])
            seq.logical_blocks = new_seq_blocks
            node.parent = node.parent.parent
            num_cum_ids = node.num_cum_ids
            seq.set_step(num_cum_ids)

        def __update_child(node: TreeNode, slice_start: int):
            """update child seq."""
            seq = node.sequence
            seq_blocks = seq.logical_blocks
            new_seq_blocks = LogicalTokenBlocks(seq_blocks[slice_start:])
            seq.logical_blocks = new_seq_blocks

        def __update_children(children: Dict[int, TreeNode], slice_start: int):
            """update children seqs."""
            for child in list(children.values()):
                if child.sequence is not None:
                    __update_child(child, slice_start)
                if len(child.children) > 0:
                    __update_children(child.children, slice_start)

        node_id = node.node_id
        assert node_id != self.root.node_id

        if node.sequence is not None:
            __update_leaf()

        num_remove_blocks = node.num_cum_blocks
        for child in list(node.children.values()):
            if child.num_blocks == 0:
                __update_empty_child(child, node.num_blocks)
            else:
                # for window attention
                child.parent = self.root
                if child.sequence is not None:
                    __update_child(child, num_remove_blocks)
                __update_children(child.children, num_remove_blocks)

        node.parent = None
        self.nodes.pop(node_id)

        device = node.device
        self._block_manager.free(node.blocks, device)

    def split_node(self, node: TreeNode, num_blocks: int):
        """split node."""
        if num_blocks > node.num_blocks or num_blocks == 0:
            return None, node

        parent = node.parent
        num_tokens = num_blocks * self.block_size
        new_node = self.new_node(token_ids=node.token_ids[:num_tokens],
                                 blocks=node.blocks[:num_blocks],
                                 parent=parent,
                                 children={node.node_id: node},
                                 device=node.device,
                                 last_visit_time=node.last_visit_time)

        node.token_ids = node.token_ids[num_tokens:]
        node.blocks = node.blocks[num_blocks:]
        node.parent = new_node
        return new_node, node

    def match_node(self, token_ids: np.ndarray):
        """match node, return first mismatch node."""

        def __match_children(node: TreeNode, token_ids: np.ndarray,
                             max_match_len: int):
            """match children."""
            best_match_node = node
            best_match_len = 0
            for child in node.children.values():
                num_child_ids = child.num_token_ids
                if num_child_ids == 0:
                    continue
                diff: np.ndarray = (
                    token_ids[:num_child_ids] == child.token_ids)
                if diff.all():
                    return __match_children(child, token_ids[num_child_ids:],
                                            max_match_len + num_child_ids)
                match_len = diff.argmin()
                if match_len > best_match_len:
                    best_match_len = match_len
                    best_match_node = child

            max_match_len += best_match_len
            if best_match_len == 0:
                best_match_len = node.num_token_ids
            return (best_match_node, best_match_len, max_match_len)

        return __match_children(self.root, token_ids, 0)

    def can_update_sequence(self, seq: SchedulerSequence):
        """can update sequence."""
        leaf_node = self.seq_node_map.get(seq, None)
        num_required = self.num_required_blocks(seq.num_token_ids, leaf_node)
        return self._block_manager.can_allocate(num_required)

    def update_sequence(self, seq: SchedulerSequence):
        """update sequence."""
        block_size = self.cache_config.block_size

        def __allocate_blocks(num_seq_full_ids: int, num_node_max_ids: int):
            """allocate blocks."""
            num_alloc_ids = max(0, num_seq_full_ids - num_node_max_ids)
            num_new_blocks = 0
            if num_alloc_ids > 0:
                num_new_blocks = _div_up(num_alloc_ids, block_size)
            return self._block_manager.allocate(num_new_blocks)

        def __fill_first_node(node: TreeNode, token_ids: np.ndarray,
                              blocks: np.ndarray):
            """fill first node."""
            num_parent_blocks = node.parent.num_cum_blocks
            num_parent_ids = node.parent.num_cum_ids
            token_ids = token_ids[num_parent_ids:]
            blocks = blocks[num_parent_blocks:]
            max_node_blocks = self.max_node_blocks
            max_node_tokens = self.max_node_tokens

            fill_token_ids = token_ids[:max_node_tokens]
            fill_blocks = blocks[:max_node_blocks]

            node.token_ids = fill_token_ids
            node.blocks = fill_blocks

            remain_token_ids = token_ids[max_node_tokens:]
            remain_blocks = blocks[max_node_blocks:]
            return remain_token_ids, remain_blocks

        node = self.seq_node_map.get(seq.seq_id, None)
        assert node is not None

        token_ids = seq.full_token_ids
        num_seq_full_ids = len(token_ids)
        num_node_cum_ids = node.num_cum_ids
        num_appendable = self.num_appendable_ids(node)

        blocks = __allocate_blocks(num_seq_full_ids,
                                   num_node_cum_ids + num_appendable)
        seq.logical_blocks.append(blocks)

        # fill first node
        token_ids, blocks = __fill_first_node(
            node, token_ids, seq.logical_blocks.get_real_blocks())

        if len(blocks) > 0:
            self.pop_map(node=node)
            node = self.add_node(node, token_ids, blocks)
            self.update_map(seq, node)

        node.update_visit_time(self.step_time, self.cache_config.window_size)

    def add_sequence(self, seq: SchedulerSequence):
        """add sequence.

        only match history and add empty node, do not allocate new node.
        """

        def __add_blocks(node: TreeNode):
            """get blocks."""
            blocks = []
            while node.node_id != self.root.node_id:
                blocks.append(node.blocks)
                node = node.parent
            reversed(blocks)
            blocks = np.concatenate(blocks)
            seq.logical_blocks.append(blocks)

        node, node_match_len, max_match_len = self.match_node(
            seq.full_token_ids)

        if node == self.root:
            # no match found, empty blocks
            parent = node
        elif node_match_len < self.block_size:
            parent = node.parent
            steps = max_match_len - node_match_len
            __add_blocks(parent)
            seq.set_step(steps)
        else:
            parent, _ = self.split_node(node,
                                        node_match_len // self.block_size)
            steps = max_match_len + parent.num_token_ids - node_match_len
            __add_blocks(parent)
            seq.set_step(steps)

        new_node = self.add_node(parent)
        self.update_map(seq, new_node)

    def remove_sequence(self, seq: SchedulerSequence):
        """remove sequence."""
        node = self.seq_node_map.get(seq.seq_id, None)
        if node is not None and node.num_blocks == 0:
            node.parent = None
            self.nodes.pop(node.node_id)
        self.pop_map(seq=seq)

    def sort_nodes(self, ignore_empty: bool = True):
        """sort nodes."""

        def __sort_key(n: TreeNode):
            """get sort key."""
            key = n.last_visit_time
            if len(n.children) == 0:
                # children first
                key -= 0.5
            return key

        nodes = list(self.nodes.values())
        nodes.remove(self.root)
        if ignore_empty:
            nodes = [node for node in nodes if node.num_blocks > 0]

        nodes = sorted(nodes, key=__sort_key)
        return nodes
