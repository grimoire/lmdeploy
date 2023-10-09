# Copyright (c) OpenMMLab. All rights reserved.
import enum
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch
from torch import Tensor

from lmdeploy.pytorch_poc.block import LogicalTokenBlock


class SamplingParam:
    """Sampling parameter."""

    def __init__(
        self,
        top_p: float = 0.8,
        top_k: int = None,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        ignore_eos: bool = False,
        random_seed: int = None,
        stop_words: List[int] = None,
        bad_words: List[int] = None,
    ):
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.ignore_eos = ignore_eos
        self.random_seed = random_seed
        self.stop_words = stop_words
        self.bad_words = bad_words


class MessageStatus(enum.Enum):
    """Status of a sequence."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAP_OUT = enum.auto()
    STOPPED = enum.auto()
    ENDED = enum.auto()
    FINISHED = enum.auto()
    ABORTED = enum.auto()


_MSG_COUNT = 0


def _new_msg_id():
    """get a new message id."""
    global _MSG_COUNT
    msg_id = _MSG_COUNT
    _MSG_COUNT += 1
    return msg_id


class SchedulerSession:
    """Scheduler session."""

    def __init__(self, session_id: int) -> None:
        self.session_id = session_id
        self.messages: Dict[SchedulerMessage] = dict()

    def add_message(
            self,
            token_ids: Tensor,
            max_output_len: int = 512,
            sampling_param: SamplingParam = None) -> 'SchedulerMessage':
        """Add a new message."""
        if sampling_param is None:
            sampling_param = SamplingParam()

        msg = SchedulerMessage(msg_id=_new_msg_id(),
                               token_ids=token_ids,
                               session=self,
                               status=MessageStatus.WAITING,
                               remain_output_len=max_output_len,
                               sampling_param=sampling_param,
                               arrive_time=time.time())
        self.messages[msg.msg_id] = msg
        return msg

    def fork_message(self,
                     token_ids: Tensor,
                     msg: 'SchedulerMessage',
                     max_output_len: int = 512) -> 'SchedulerMessage':
        """Fork a new message from exist message."""
        assert msg.session == self

        new_msg = SchedulerMessage(
            msg_id=_new_msg_id(),
            token_ids=token_ids,
            session=self,
            history_token_ids=msg.history_token_ids.clone(),
            status=msg.status,
            remain_output_len=max_output_len,
            logical_blocks=deepcopy(msg.logical_blocks),
            sampling_param=deepcopy(msg.sampling_param),
            arrive_time=time.time(),
            meta=deepcopy(msg.meta))

        self.messages[new_msg.msg_id] = new_msg
        return new_msg


@dataclass
class SchedulerMessage:
    """Scheduler message."""
    msg_id: int
    token_ids: Tensor
    session: SchedulerSession
    history_token_ids: Tensor = field(default=torch.empty(0, dtype=torch.long))
    status: MessageStatus = MessageStatus.WAITING
    remain_output_len: int = 0
    logical_blocks: Sequence[LogicalTokenBlock] = field(default_factory=list)
    sampling_param: SamplingParam = field(default_factory=SamplingParam)
    arrive_time: float = 0.0
    meta: Any = None

    @property
    def history_len(self):
        return len(self.history_token_ids)

    def append_tokens(self, num_tokens: int, block_size: int):
        """Append new tokens, update logical blocks.

        Args:
            num_tokens (int): Number of tokens.
            block_size (int): Size of block.
        """
        if len(self.logical_blocks) == 0:
            remain_num_tokens = num_tokens
            next_block_id = 0
        else:
            last_block = self.logical_blocks[-1]
            num_empty_slots = last_block.get_num_empty_slots()
            num_append_slots = min(num_tokens, num_empty_slots)
            last_block.append_tokens(num_append_slots)
            remain_num_tokens = num_tokens - num_append_slots
            next_block_id = last_block.block_id + 1

        for block_id_offset, msg_offset in enumerate(
                range(0, remain_num_tokens, block_size)):
            num_tokens = min(remain_num_tokens - msg_offset, block_size)
            logical_block = LogicalTokenBlock(next_block_id + block_id_offset,
                                              block_size)
            logical_block.append_tokens(num_tokens=num_tokens)
            self.logical_blocks.append(logical_block)
