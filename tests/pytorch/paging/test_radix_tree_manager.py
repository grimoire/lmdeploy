import numpy as np
import pytest

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.messages import SchedulerSession
from lmdeploy.pytorch.paging.block_manager import build_block_manager
from lmdeploy.pytorch.paging.radix_tree_manager import (RadixTreeManager,
                                                        TreeNode)


def _np_randint(num: int):
    return np.random.randint(0, 100, (num, ))


def _create_seq(session: SchedulerSession,
                num_tokens: int = 0,
                num_history: int = 0):
    seq = session.add_sequence(_np_randint(num_history))
    seq.update_token_ids(_np_randint(num_tokens))
    return seq


class TestTreeNode:

    @pytest.fixture
    def cache_config(self):
        yield CacheConfig(block_size=16,
                          num_cpu_blocks=4,
                          num_gpu_blocks=32,
                          window_size=-1)

    @pytest.fixture
    def block_manager(self, cache_config):
        yield build_block_manager(cache_config)

    @pytest.fixture
    def rtree_manager(self, block_manager, cache_config):
        yield RadixTreeManager(cache_config, block_manager)

    def test_node(self, rtree_manager):
        root = TreeNode(node_id=0,
                        manager=rtree_manager,
                        token_ids=_np_randint(64),
                        blocks=_np_randint(4),
                        parent=None)

        assert root.num_blocks == 4
        assert root.num_token_ids == 64

        child0 = TreeNode(
            node_id=1,
            manager=rtree_manager,
            token_ids=_np_randint(32),
            blocks=_np_randint(2),
            parent=root,
        )

        child1 = TreeNode(
            node_id=2,
            manager=rtree_manager,
            token_ids=_np_randint(48),
            blocks=_np_randint(3),
            parent=root,
        )

        # test update parent
        assert len(root.children) == 2

        # test cum
        assert child0.num_cum_ids == 96
        assert child0.num_cum_blocks == 6
        assert child1.num_cum_ids == 112
        assert child1.num_cum_blocks == 7

        # test change parent
        child1.parent = child0
        assert len(root.children) == 1
        assert len(child0.children) == 1

        # test visit time
        child1.update_visit_time(4)
        assert root.last_visit_time == 4
        assert child0.last_visit_time == 4
        assert child1.last_visit_time == 4

        child1.update_visit_time(5, window_size=48)
        assert root.last_visit_time == 4
        assert child0.last_visit_time == 4
        assert child1.last_visit_time == 5

        child1.update_visit_time(6, window_size=50)
        assert root.last_visit_time == 4
        assert child0.last_visit_time == 6
        assert child1.last_visit_time == 6

        child1.update_visit_time(7, window_size=90)
        assert root.last_visit_time == 7
        assert child0.last_visit_time == 7
        assert child1.last_visit_time == 7


class TestRadixTreeManager:

    @pytest.fixture
    def window_size(self, request):
        yield request.param

    @pytest.fixture
    def cache_config(self, window_size):
        yield CacheConfig(block_size=16,
                          num_cpu_blocks=4,
                          num_gpu_blocks=512,
                          window_size=window_size)

    @pytest.fixture
    def block_manager(self, cache_config):
        yield build_block_manager(cache_config)

    @pytest.fixture
    def rtree_manager(self, block_manager, cache_config):
        yield RadixTreeManager(cache_config, block_manager)

    @pytest.mark.parametrize('window_size', [-1, 32])
    def test_add_node(self, cache_config, rtree_manager):
        # test empty
        node0 = rtree_manager.add_node()
        assert len(rtree_manager.nodes) == 2

        # test small node
        node1 = rtree_manager.add_node(node0,
                                       token_ids=_np_randint(4),
                                       blocks=_np_randint(1))
        assert len(rtree_manager.nodes) == 3
        assert node1.parent == node0

        if cache_config.window_size > 0:
            num_hidden = 5
            window_size = cache_config.window_size
            block_per_window = window_size // cache_config.block_size
            num_tokens = window_size * num_hidden + 1
            num_blocks = block_per_window * num_hidden + 1
            node2 = rtree_manager.add_node(token_ids=_np_randint(num_tokens),
                                           blocks=_np_randint(num_blocks))
            assert len(rtree_manager.nodes) == 3 + num_hidden + 1
            count = 0
            parent = node2.parent
            while parent != rtree_manager.root:
                parent = parent.parent
                count += 1
            assert count == num_hidden

    @pytest.mark.parametrize('window_size', [-1])
    def test_split_node(self, cache_config, rtree_manager):
        block_size = cache_config.block_size
        num_blocks = 4
        token_ids = _np_randint(num_blocks * block_size)
        blocks = _np_randint(num_blocks)
        node0 = rtree_manager.add_node(token_ids=token_ids, blocks=blocks)

        # test invalid split
        spnode0, spnode1 = rtree_manager.split_node(node0, 0)
        assert spnode0 is None
        assert spnode1 == node0
        assert len(rtree_manager.nodes) == 2

        spnode0, spnode1 = rtree_manager.split_node(node0, num_blocks + 1)
        assert spnode0 is None
        assert spnode1 == node0
        assert len(rtree_manager.nodes) == 2

        # test normal
        spnode0, spnode1 = rtree_manager.split_node(node0, 1)
        assert spnode0.num_blocks == 1
        assert spnode0.num_token_ids == block_size
        assert spnode1 == node0
        assert spnode1.num_blocks == num_blocks - 1
        assert spnode1.num_token_ids == block_size * (num_blocks - 1)
        assert len(rtree_manager.nodes) == 3

        num_blocks = num_blocks - 1
        spnode0, spnode1 = rtree_manager.split_node(spnode1, num_blocks)
        assert spnode0.num_blocks == num_blocks
        assert spnode0.num_token_ids == block_size * num_blocks
        assert spnode1 == node0
        assert spnode1.num_blocks == 0
        assert spnode1.num_token_ids == 0

    @pytest.mark.parametrize('window_size', [-1])
    def test_match_node(self, cache_config, rtree_manager):
        block_size = cache_config.block_size
        token_ids = _np_randint(block_size)

        # test no match
        rtree_manager.add_node(token_ids=token_ids + 1, blocks=_np_randint(1))
        output = rtree_manager.match_node(token_ids)
        match_node, best_match_len, max_match_len = output
        assert match_node == rtree_manager.root
        assert best_match_len == 0
        assert max_match_len == 0

        # test no match empty child
        node0 = rtree_manager.add_node(token_ids=token_ids,
                                       blocks=_np_randint(1))
        rtree_manager.add_node(node0)
        output = rtree_manager.match_node(token_ids)
        match_node, best_match_len, max_match_len = output
        assert match_node == node0
        assert best_match_len == len(token_ids)
        assert max_match_len == len(token_ids)

        # test match partial child
        child_ids = _np_randint(block_size)
        child1 = rtree_manager.add_node(node0,
                                        token_ids=child_ids,
                                        blocks=_np_randint(1))
        token_ids = np.concatenate([token_ids, child_ids])
        token_ids[block_size + block_size // 2:] += 1
        output = rtree_manager.match_node(token_ids)
        match_node, best_match_len, max_match_len = output
        assert match_node == child1
        assert best_match_len == block_size // 2
        assert max_match_len == block_size + block_size // 2

        # test child > token len

    @pytest.mark.parametrize('window_size', [-1])
    def test_add_sequence(self, cache_config, rtree_manager):
        block_size = cache_config.block_size
        session = SchedulerSession(1, block_size=cache_config.block_size)
        seq = _create_seq(session, block_size * 5 + 1)

        # test no match
        rtree_manager.add_sequence(seq)
        assert len(rtree_manager.nodes) == 2
        node = rtree_manager.seq_node_map[seq.seq_id]
        assert node.num_token_ids == 0
        assert node.num_blocks == 0

        # test match less than one block
        seq = _create_seq(session, block_size * 5 + 1)
        node0 = rtree_manager.add_node(
            token_ids=seq.full_token_ids[:block_size * 5],
            blocks=_np_randint(5))
        rtree_manager.add_node(node0,
                               token_ids=seq.full_token_ids[block_size * 5:],
                               blocks=_np_randint(1))
        nodes_count = len(rtree_manager.nodes)
        rtree_manager.add_sequence(seq)
        assert len(rtree_manager.nodes) == nodes_count + 1
        assert len(node0.children) == 2
        assert seq.num_history_ids == block_size * 5

        # test normal
        seq = _create_seq(session, block_size * 5 + 1)
        node0 = rtree_manager.add_node(token_ids=seq.full_token_ids,
                                       blocks=_np_randint(6))
        nodes_count = len(rtree_manager.nodes)
        rtree_manager.add_sequence(seq)
        assert len(rtree_manager.nodes) == nodes_count + 2
        assert len(node0.parent.children) == 2
        assert node0.num_token_ids == 1
        assert node0.num_blocks == 1
        assert node0.parent.num_token_ids == block_size * 5
        assert node0.parent.num_blocks == 5
        assert seq.num_history_ids == block_size * 5

    @pytest.mark.parametrize('window_size', [32, -1])
    def test_update_sequence(self, cache_config, block_manager, rtree_manager):
        block_size = cache_config.block_size
        session = SchedulerSession(1, block_size=cache_config.block_size)

        # test add first
        seq = _create_seq(session, block_size + 1)
        rtree_manager.add_sequence(seq)
        rtree_manager.update_sequence(seq)
        node = rtree_manager.seq_node_map[seq.seq_id]
        assert len(rtree_manager.nodes) == 2
        assert block_manager.get_num_free_gpu_blocks(
        ) == cache_config.num_gpu_blocks - 2
        assert node.num_token_ids == block_size + 1
        assert node.num_blocks == 2

        # test append
        seq.update_token_ids(_np_randint(block_size + 1))
        rtree_manager.update_sequence(seq)
        node = rtree_manager.seq_node_map[seq.seq_id]
        if cache_config.window_size == 32:
            assert len(rtree_manager.nodes) == 3
            assert block_manager.get_num_free_gpu_blocks(
            ) == cache_config.num_gpu_blocks - 3
            assert node.num_token_ids == 2
            assert node.num_blocks == 1
            assert node.parent.num_token_ids == 32
            assert node.parent.num_blocks == 2
        else:
            num_blocks = 3
            assert len(rtree_manager.nodes) == 2
            assert block_manager.get_num_free_gpu_blocks(
            ) == cache_config.num_gpu_blocks - num_blocks
            assert node.num_token_ids == block_size * 2 + 2
            assert node.num_blocks == num_blocks

    @pytest.mark.parametrize('window_size', [-1])
    def test_remove_sequence(self, cache_config, rtree_manager):
        block_size = cache_config.block_size
        session = SchedulerSession(1, block_size=cache_config.block_size)
        seq = _create_seq(session, block_size * 3)
        rtree_manager.add_sequence(seq)
        rtree_manager.update_sequence(seq)
        rtree_manager.remove_sequence(seq)
        assert len(rtree_manager.nodes) == 2
        assert len(rtree_manager.seq_node_map) == 0

    @pytest.mark.parametrize('window_size', [-1])
    def test_remove_node(self, cache_config, block_manager, rtree_manager):
        block_size = cache_config.block_size
        session = SchedulerSession(1, block_size=cache_config.block_size)

        # test remove leaf
        seq = _create_seq(session, block_size * 3)
        rtree_manager.add_sequence(seq)
        rtree_manager.update_sequence(seq)
        node0 = rtree_manager.seq_node_map[seq.seq_id]
        _, node0 = rtree_manager.split_node(node0, 2)

        rtree_manager.remove_node(node0)
        expected_node_count = 3
        node_count = len(rtree_manager.nodes)
        expected_free_blocks = cache_config.num_gpu_blocks - 2
        free_blocks = block_manager.get_num_free_gpu_blocks()
        assert node_count == expected_node_count
        assert free_blocks == expected_free_blocks
        assert node0.sequence is None
        assert seq.seq_id in rtree_manager.seq_node_map
        seq_node = rtree_manager.seq_node_map[seq.seq_id]
        assert seq_node.num_token_ids == 0
        assert seq_node.num_blocks == 0
        assert seq.num_history_ids == block_size * 2
        assert seq.num_token_ids == block_size
        assert len(seq.logical_blocks) == 2

        # test remove parent of empty child
        seq = _create_seq(session, block_size * 3)
        rtree_manager.add_sequence(seq)
        rtree_manager.update_sequence(seq)
        node0 = rtree_manager.seq_node_map[seq.seq_id]
        prev_node, node0 = rtree_manager.split_node(node0, 3)
        assert node0.num_blocks == 0
        prev_node0, prev_node1 = rtree_manager.split_node(prev_node, 2)
        # root -> prev_node0 -> prev_node1 -> node0(empty)
        # 3 new nodes, remove prev_node1
        rtree_manager.remove_node(prev_node1)

        expected_node_count = node_count + 2
        node_count = len(rtree_manager.nodes)
        expected_free_blocks = free_blocks - 2
        free_blocks = block_manager.get_num_free_gpu_blocks()
        assert node_count == expected_node_count
        assert free_blocks == expected_free_blocks
        assert node0.parent == prev_node0
        assert node0.sequence == seq
        assert seq.num_history_ids == block_size * 2
        assert len(seq.logical_blocks) == 2

        # test remove non-empty child
        seq = _create_seq(session, block_size * 3)
        rtree_manager.add_sequence(seq)
        rtree_manager.update_sequence(seq)
        seq.update_token_ids([])
        node0 = rtree_manager.seq_node_map[seq.seq_id]
        prev_node, node0 = rtree_manager.split_node(node0, 2)
        prev_node0, prev_node1 = rtree_manager.split_node(prev_node, 1)
        # root -> prev_node0 -> prev_node1 -> node0(1 block)
        # 3 new nodes, remove prev_node1
        rtree_manager.remove_node(prev_node1)
        expected_node_count = node_count + 2
        node_count = len(rtree_manager.nodes)
        expected_free_blocks = free_blocks - 2
        free_blocks = block_manager.get_num_free_gpu_blocks()
        assert node_count == expected_node_count
        assert free_blocks == expected_free_blocks
        assert node0.parent == rtree_manager.root
        assert node0.sequence == seq
        assert seq.num_history_ids == block_size * 3
        assert len(seq.logical_blocks) == 1
        real_blocks = seq.logical_blocks.get_real_blocks()
        np.testing.assert_allclose(real_blocks, node0.blocks)

    @pytest.mark.parametrize('window_size', [-1])
    def test_sort_nodes(self, cache_config, rtree_manager):
        # test sort
        block_size = cache_config.block_size
        session = SchedulerSession(1, block_size=cache_config.block_size)
        seq = _create_seq(session, block_size * 3)
        rtree_manager.add_sequence(seq)
        rtree_manager.update_sequence(seq)
        seq_node = rtree_manager.seq_node_map[seq.seq_id]
        prev_node, seq_node = rtree_manager.split_node(seq_node, 2)
        prev_node0, prev_node1 = rtree_manager.split_node(prev_node, 1)
        seq_node.update_visit_time(2, block_size * 2)
        prev_node1 = seq_node.parent
        prev_node0 = prev_node1.parent

        sorted_nodes = rtree_manager.sort_nodes()
        assert sorted_nodes[0] == prev_node0
        assert sorted_nodes[1] == seq_node
        assert sorted_nodes[2] == prev_node1
