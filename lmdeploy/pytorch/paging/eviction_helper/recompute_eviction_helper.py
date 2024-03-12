# Copyright (c) OpenMMLab. All rights reserved.
from ...adapter.adapter import SchedulerAdapter
from ...messages import SchedulerSequence
from .base_eviction_helper import BaseEvictionHelper


class RecomputeEvictionHelper(BaseEvictionHelper):
    """recompute eviction."""

    def num_seq_required_blocks(self, seq: SchedulerSequence):
        """num seq required blocks."""
        return self.rtree_manager.num_seq_required_blocks(seq)

    def try_update_sequence(self,
                            eviction_nodes,
                            seq: SchedulerSequence,
                            adapter: SchedulerAdapter = None):
        """try evict one non-empty node."""
        num_required = self.num_seq_required_blocks(seq)
        if adapter is not None:
            num_ada_required = self.num_adapter_required_blocks(adapter)
            num_required += num_ada_required

        num_free_blocks = self.block_manager.get_num_free_gpu_blocks()
        num_required -= num_free_blocks

        can_alloc = True
        step_time = self.rtree_manager.step_time
        seq_nodes = []
        if num_required > 0:
            seq_node = self.rtree_manager.seq_node_map[seq.seq_id]
            while seq_node != self.rtree_manager.root:
                seq_nodes.append(seq_node)
                seq_node = seq_node.parent

        while num_required > 0:
            if len(eviction_nodes) == 0:
                can_alloc = False
                break

            node = eviction_nodes.pop(0)
            if node.last_visit_time >= step_time:
                continue

            if node in seq_nodes:
                continue

            num_blocks = node.num_blocks
            self.rtree_manager.remove_node(node)
            num_required -= num_blocks

        if can_alloc:
            self.rtree_manager.update_sequence(seq)

        return can_alloc
