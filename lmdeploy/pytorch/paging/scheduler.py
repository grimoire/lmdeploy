# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Set, Union

from lmdeploy.utils import get_logger, logging_timer

from ..adapter.adapter import ADAPTER_MANAGER, SchedulerAdapter
from ..config import CacheConfig, SchedulerConfig
from ..messages import (MessageStatus, SchedulerSequence, SchedulerSession,
                        SequenceManager)
from .block_manager import build_block_manager
from .radix_tree_manager import RadixTreeManager

logger = get_logger('lmdeploy')

SeqList = List[SchedulerSequence]
AdapterList = List[SchedulerAdapter]


def _find_seq_with_session_id(group: SeqList, session_id: int):
    return [seq for seq in group if seq.session_id == session_id]


@dataclass
class SchedulerOutput:
    """Output of schedule."""

    running: SeqList
    swap_in_map: Dict[int, int]
    swap_out_map: Dict[int, int]
    copy_map: Dict[int, int]
    adapters: AdapterList


class Scheduler:
    """Tools to schedule next step.

    Args:
        scheduler_config (SchedulerConfig): The config of scheduler.
        cache_config (CacheConfig): The config of cache info.
    """

    def __init__(self, scheduler_config: SchedulerConfig,
                 cache_config: CacheConfig) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        self.sessions: Dict[int, SchedulerSession] = OrderedDict()
        self.actived_adapters: Set[str] = set()

        self.block_manager = build_block_manager(cache_config)
        self.rtree_manager = RadixTreeManager(cache_config, self.block_manager)

        self.eviction_helper = self.build_eviction_helper(
            self.scheduler_config.eviction_type)

        self.seq_manager = SequenceManager()

    @property
    def waiting(self):
        """get waiting sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.WAITING)
        return list(seq_map.values())

    @property
    def running(self):
        """get waiting sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.RUNNING)
        return list(seq_map.values())

    @property
    def hanging(self):
        """get waiting sequence."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.STOPPED)
        return list(seq_map.values())

    def build_eviction_helper(self, eviction_type: str):
        if eviction_type == 'copy':
            logger.warning('Copy eviction has been deprecated. '
                           'Swap to recompute eviction.')
            eviction_type = 'recompute'

        if eviction_type == 'recompute':
            from .eviction_helper import RecomputeEvictionHelper
            return RecomputeEvictionHelper(self)
        else:
            raise TypeError(f'Unknown eviction type: {eviction_type}')

    def _set_message_status(self, message: SchedulerSequence,
                            status: MessageStatus):
        """Set status of message.

        Args:
            message (SchedulerSequence): message to setup status.
            status (MessageStatus): New message status.
        """
        message.status = status

    def add_session(self, session_id: int):
        """Add new session.

        Args:
            session_id (int): New session id.
        """
        assert session_id not in self.sessions
        session = SchedulerSession(session_id,
                                   self.cache_config.block_size,
                                   seq_manager=self.seq_manager)
        self.sessions[session_id] = session
        return session

    def add_sequence(self, seq: SchedulerSequence):
        """Add sequence.

        Args:
            seq (SchedulerSequence): New sequence.
        """
        assert (seq.session_id
                in self.sessions), f'Unknown session id {seq.session_id}'

        # push message to waiting queue
        self._set_message_status(seq, MessageStatus.WAITING)
        self.rtree_manager.add_sequence(seq)

    def add_adapter(self, adapter_path: str, adapter_name: str):
        """Add adapter.

        Args:
            adapter_path (str): The path of adapter.
            adapter_name (str): The name of the adapter.
        """

        def __allocate_adapter(adapter: SchedulerAdapter):
            """allocate adapter."""
            num_required_blocks = 0
            if not adapter.is_actived():
                num_required_blocks = adapter.rank * len(
                    adapter.target_modules)
            if num_required_blocks > 0:
                blocks = self.block_manager.allocate(num_required_blocks,
                                                     'cpu')
                adapter.logical_blocks.append(blocks)

        adapter = ADAPTER_MANAGER.add_adapter_from_pretrained(
            adapter_path, adapter_name=adapter_name)
        __allocate_adapter(adapter)
        logical_blocks = adapter.logical_blocks.get_real_blocks()
        block_table = self.block_manager.get_physical_blocks(
            logical_blocks) - self.block_manager.num_gpu_blocks
        return adapter.build_weight_map(block_table)

    @logging_timer('SchedulePrefilling', logger)
    def _schedule_prefill(self):
        """Schedule for prefilling."""
        max_batches = self.scheduler_config.max_batches - len(self.running)
        eviction_helper = self.eviction_helper
        swap_out_map: Dict[int, int] = dict()
        swap_in_map: Dict[int, int] = dict()
        copy_map: Dict[int, int] = dict()
        running: SeqList = []
        required_adapters = set(seq.adapter_name for seq in self.running)
        max_adapters = self.scheduler_config.max_active_adapters - len(
            required_adapters)
        eviction_nodes = self.rtree_manager.sort_nodes()

        def _to_running(seq: SchedulerSequence):
            """to running."""
            self._set_message_status(seq, MessageStatus.RUNNING)
            running.append(seq)

        def __try_update_sequence(seq: SchedulerSequence):
            """evict until can append."""
            adapter = ADAPTER_MANAGER.get_adapter(seq.adapter_name)
            return eviction_helper.try_update_sequence(eviction_nodes, seq,
                                                       adapter)

        def _reorder_waiting():
            """reorder waiting."""
            return sorted(self.waiting, key=lambda seq: seq.arrive_time)

        def _active_adapter(adapter_name):
            """active adapter of a seq."""
            if adapter_name in required_adapters:
                return
            if adapter_name is None:
                required_adapters.add(adapter_name)
                return
            adapter = ADAPTER_MANAGER.get_adapter(adapter_name)
            if not adapter.is_actived():
                logical_blocks = adapter.logical_blocks.get_real_blocks()
                tmp_map = self.block_manager.swap_in(logical_blocks)
                swap_in_map.update(tmp_map)
                adapter.active(True)
            required_adapters.add(adapter_name)

        def _deactive_adapter(adapter_name):
            """deactive_adapter."""
            if adapter_name is None:
                return
            adapter = ADAPTER_MANAGER.get_adapter(adapter_name)
            if adapter.is_actived():
                tmp_map = self.block_manager.swap_out(adapter)
                swap_out_map.update(tmp_map)
                adapter.active(False)

        if len(running) >= max_batches or len(self.waiting) == 0:
            return running, swap_in_map, swap_out_map, copy_map

        # update visit time of all running sequence
        step_time = self.rtree_manager.step_time
        for seq in self.running:
            node = self.rtree_manager.seq_node_map[seq.seq_id]
            node.update_visit_time(step_time)

        waiting = _reorder_waiting()
        while len(waiting) > 0 and len(running) < max_batches:
            seq = waiting[0]

            # limit number of adapters
            if (len(required_adapters) >= max_adapters
                    and seq.adapter_name not in required_adapters):
                break

            if not __try_update_sequence(seq):
                break
            _active_adapter(seq.adapter_name)
            waiting.pop(0)
            _to_running(seq)

        deactive_adapters = self.actived_adapters.difference(required_adapters)
        for adapter_name in deactive_adapters:
            _deactive_adapter(adapter_name)
        self.actived_adapters = required_adapters

        return running, swap_in_map, swap_out_map, copy_map

    @logging_timer('ScheduleDecoding', logger)
    def _schedule_decoding(self):
        """schedule decoding."""
        assert len(self.running) != 0

        eviction_helper = self.eviction_helper
        swap_out_map: Dict[int, int] = dict()
        swap_in_map: Dict[int, int] = dict()
        copy_map: Dict[int, int] = dict()
        eviction_nodes = self.rtree_manager.sort_nodes()

        def __try_update_sequence(seq: SchedulerSequence):
            """evict until can append."""
            return eviction_helper.try_update_sequence(eviction_nodes, seq)

        # 1. running
        for seq in self.running:
            # token + 1

            if len(seq.logical_blocks) > self.block_manager.num_gpu_blocks:
                # Reach max gpu cache size.
                logger.warning(f'session[{seq.session_id}] '
                               f'sequence[{seq.seq_id}] '
                               'reach max gpu size.')
                self._set_message_status(seq, MessageStatus.ABORTED)
                continue

            if seq.num_token_ids > 1:
                self._set_message_status(seq, MessageStatus.WAITING)
                continue

            if not __try_update_sequence(seq):
                self._set_message_status(seq, MessageStatus.WAITING)

        return self.running, swap_in_map, swap_out_map, copy_map

    @classmethod
    def _get_adapter_list(cls, adapter_names: List[str]):
        adapters = [
            ADAPTER_MANAGER.get_adapter(name) for name in adapter_names
        ]
        return adapters

    def schedule(self, is_prefill: bool):
        """Schedule inputs for next steps."""
        self.rtree_manager.add_step_time()
        if is_prefill:
            output = self._schedule_prefill()
        else:
            output = self._schedule_decoding()
        running, swap_in_map, swap_out_map, copy_map = output

        adapters = self._get_adapter_list(self.actived_adapters)

        return SchedulerOutput(running=running,
                               swap_in_map=swap_in_map,
                               swap_out_map=swap_out_map,
                               copy_map=copy_map,
                               adapters=adapters)

    def _set_session_status(self, session_id: int, status: MessageStatus):
        """Setup the status of session.

        Args:
            session_id (int): The session id.
            status (MessageStatus): New status.
        """
        assert session_id in self.sessions
        session = self.sessions[session_id]
        session.status = status
        for seq in session.sequences.values():
            seq.status = status

    def stop_session(self, session_id: int):
        """Stop session.

        Args:
            session_id (int): The session id.
        """
        self._set_session_status(session_id, MessageStatus.STOPPED)

    def end_session(self, session_id: int):
        """End session.

        Args:
            session_id (int): The session id.
        """
        self._set_session_status(session_id, MessageStatus.ENDED)

    def has_unfinished(self):
        """Check if there are any unfinished message."""
        return self.has_running() or len(self.waiting) > 0

    def has_running(self):
        return len(self.running) > 0

    def _remove_sequence(self, seq: SchedulerSequence):
        """Remove sequence(unsafe)

        Args:
            seq (SchedulerSequence): sequence to remove
        """
        self.rtree_manager.remove_sequence(seq)
        seq.session.remove_sequence(seq)

    def _remove_session(self, session: SchedulerSession):
        """remove session."""
        for seq in session.sequences.values():
            self._remove_sequence(seq)
        self.sessions.pop(session.session_id)

    def update(self):
        """Update scheduler status after one step.

        A full step inference should include:
        0. end unused sequence
        1. schedule the running sequence
        2. forward with the running sequence
        3. update scheduler status
        """
        seq_to_remove = self.seq_manager.get_sequences(MessageStatus.ENDED)
        seq_to_remove = list(seq_to_remove.values())

        # remove seqs
        for seq in seq_to_remove:
            self._remove_sequence(seq)

        # remove session
        for session in list(self.sessions.values()):
            if session.status == MessageStatus.ENDED:
                self._remove_session(session)

    def get_block_tables(self, seqs: Union[SeqList, AdapterList]):
        """get block table of the sequences."""
        return [
            self.block_manager.get_physical_blocks(
                seq.logical_blocks.get_real_blocks()) for seq in seqs
        ]
