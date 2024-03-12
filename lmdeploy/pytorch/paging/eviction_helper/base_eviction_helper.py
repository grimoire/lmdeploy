# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from ...adapter.adapter import SchedulerAdapter
from ...messages import SchedulerSequence
from ..scheduler import Scheduler

SeqList = List[SchedulerSequence]


class BaseEvictionHelper:
    """Base eviction helper."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler: Scheduler = scheduler
        self.block_manager = scheduler.block_manager
        self.rtree_manager = scheduler.rtree_manager
        self.cache_config = self.scheduler.cache_config

    def num_seq_required_blocks(self, seq: SchedulerSequence):
        """num seq required blocks."""
        num_token_ids = seq.num_token_ids
        seq_node = self.rtree_manager.seq_node_map[seq.seq_id]
        return self.rtree_manager.num_required_blocks(num_token_ids, seq_node)

    def num_adapter_required_blocks(self, adapter: SchedulerAdapter):
        """num adapter required blocks."""
        if adapter.is_actived():
            return 0
        return len(adapter.logical_blocks)

    def try_update_sequence(self,
                            eviction_nodes,
                            seq: SchedulerSequence,
                            adapter: SchedulerAdapter = None):
        """try evict one non-empty node."""
        raise NotImplementedError('Not implemented.')
