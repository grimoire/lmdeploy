# Copyright (c) OpenMMLab. All rights reserved.
import enum
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from torch import Tensor

from lmdeploy.messages import EngineGenerationConfig
from lmdeploy.utils import get_logger

from .block import LogicalTokenBlocks

logger = get_logger('lmdeploy')


@dataclass
class SamplingParam:
    """Sampling parameter."""
    top_p: float = 1.0
    top_k: int = 1
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    random_seed: int = None
    stop_words: List[int] = field(default_factory=list)
    bad_words: List[int] = field(default_factory=list)
    max_new_tokens: int = 512
    min_new_tokens: int = 0

    def logical_sampling_param(self):
        """create a SamplingParam for logical sampling."""
        return SamplingParam(top_p=self.top_p,
                             top_k=self.top_k,
                             temperature=self.temperature,
                             repetition_penalty=self.repetition_penalty,
                             ignore_eos=self.ignore_eos,
                             random_seed=self.random_seed,
                             bad_words=self.bad_words)

    @classmethod
    def from_gen_config(self, gen_config: EngineGenerationConfig):
        """from gen config."""
        min_new_tokens = gen_config.min_new_tokens or 0

        stop_words = gen_config.stop_words or []
        bad_words = gen_config.bad_words or []
        if gen_config.ignore_eos:
            bad_words += stop_words

        top_k = gen_config.top_k
        top_p = gen_config.top_p
        temperature = gen_config.temperature
        repetition_penalty = gen_config.repetition_penalty
        max_new_tokens = gen_config.max_new_tokens

        if top_k <= 0:
            logger.warning('`top_k` has to be a strictly'
                           f' positive value, but is {top_k}')
            top_k = 1
        if top_p < 0 or top_p > 1.0:
            logger.warning('`top_p` has to be a float > 0 and < 1'
                           f' but is {top_p}')
            top_p = 1.0
        if temperature <= 0:
            logger.warning('`temperature` has to be a strictly'
                           f' positive value, but is {temperature}')
            temperature = 1.0
        if repetition_penalty <= 0:
            logger.warning('`repetition_penalty` has to be a strictly'
                           f' positive value, but is {repetition_penalty}')
            repetition_penalty = 1.0
        if max_new_tokens < 0:
            logger.warning('`max_new_tokens` has to be a strictly'
                           f' positive value, but is {max_new_tokens}')
            max_new_tokens = 512
        if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
            logger.warning('`min_new_tokens` has to be '
                           'a int >=0 and <= `max_new_tokens`,'
                           f' but is {min_new_tokens}')
            min_new_tokens = 0
        return SamplingParam(top_p=top_p,
                             top_k=top_k,
                             temperature=temperature,
                             repetition_penalty=repetition_penalty,
                             ignore_eos=gen_config.ignore_eos,
                             random_seed=gen_config.random_seed,
                             stop_words=stop_words,
                             bad_words=bad_words,
                             max_new_tokens=max_new_tokens,
                             min_new_tokens=min_new_tokens)


class MessageStatus(enum.Enum):
    """Status of a sequence."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    STOPPED = enum.auto()
    ENDED = enum.auto()
    ABORTED = enum.auto()


_SEQ_COUNT = 0


def _new_msg_id():
    """get a new message id."""
    global _SEQ_COUNT
    seq_id = _SEQ_COUNT
    _SEQ_COUNT += 1
    return seq_id


class SchedulerSession:
    """Scheduler session."""

    def __init__(self, session_id: int, block_size: int) -> None:
        self.session_id = session_id
        self.block_size = block_size
        self.status: MessageStatus = MessageStatus.RUNNING
        self.sequences: Dict[int, SchedulerSequence] = dict()

    def add_sequence(self,
                     token_ids: Tensor,
                     sampling_param: SamplingParam = None,
                     adapter_name: str = None,
                     return_logits: bool = False) -> 'SchedulerSequence':
        """Add a new message."""
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.numpy()
        elif not isinstance(token_ids, np.ndarray):
            token_ids = np.array(token_ids)
        if token_ids.ndim == 0:
            token_ids = token_ids.unsqueeze(0)
        if sampling_param is None:
            sampling_param = SamplingParam()

        seq = SchedulerSequence(seq_id=_new_msg_id(),
                                session=self,
                                history_cache=HistoryTokenIds(token_ids),
                                status=MessageStatus.WAITING,
                                num_new_tokens=0,
                                sampling_param=sampling_param,
                                adapter_name=adapter_name,
                                arrive_time=time.time(),
                                return_logits=return_logits)
        self.sequences[seq.seq_id] = seq
        return seq

    def fork_sequence(
            self,
            seq: 'SchedulerSequence',
            sampling_param: SamplingParam = None) -> 'SchedulerSequence':
        """Fork a new message from exist message."""
        if sampling_param is None:
            sampling_param = deepcopy(seq.sampling_param)
        assert seq.session == self

        new_msg = SchedulerSequence(seq_id=_new_msg_id(),
                                    session=self,
                                    history_cache=seq.history_cache.clone(),
                                    num_new_tokens=0,
                                    sampling_param=sampling_param,
                                    status=seq.status,
                                    logical_blocks=seq.logical_blocks.clone(),
                                    adapter_name=seq.adapter_name,
                                    arrive_time=time.time(),
                                    meta=deepcopy(seq.meta),
                                    return_logits=seq.return_logits,
                                    random_offsets=seq.random_offsets + 1)
        new_msg._history_len = seq._history_len
        new_msg._tokens_len = seq._tokens_len

        self.sequences[new_msg.seq_id] = new_msg
        return new_msg


def _div_up(x, n):
    """perform div up."""
    return (x + n - 1) // n


def _round_up(x, n):
    """perform round up."""
    return _div_up(x, n) * n


class HistoryTokenIds:
    """history token ids."""
    ALLOC_SIZE = 1024

    def __init__(self, token_ids: np.ndarray = None):
        if token_ids is None:
            self._token_ids = np.empty((self.ALLOC_SIZE, ), dtype=np.int64)
            self._num_real = 0
        else:
            self._token_ids = token_ids
            self._num_real = len(token_ids)

    def reserve(self, size: int):
        """reserve cache."""
        num_tokens = len(self._token_ids)
        if num_tokens >= size:
            return
        reserve_size = _round_up(size - num_tokens, self.ALLOC_SIZE)
        new_token_ids = np.pad(self._token_ids, (0, reserve_size))
        self._token_ids = new_token_ids

    def get_real(self):
        """get logical blocks."""
        return self._token_ids[:self._num_real]

    def __setitem__(self, *args, **kwargs):
        """set values."""
        return self.get_real().__setitem__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        """get values."""
        return self.get_real().__getitem__(*args, **kwargs)

    def append(self, token_ids: np.ndarray):
        """append token ids."""
        num_tokens = len(token_ids)
        self.reserve(num_tokens + self._num_real)
        slice_start = self._num_real
        slice_end = slice_start + num_tokens
        self._num_real += num_tokens
        self.__setitem__(slice(slice_start, slice_end), token_ids)

    def __len__(self):
        """get length."""
        return self._num_real

    def clone(self):
        """clone."""
        ret = HistoryTokenIds()
        ret.append(self[...])
        return ret

    def copy(self):
        """copy."""
        return self.clone()


@dataclass
class SchedulerSequence:
    """Scheduler message."""
    seq_id: int
    session: SchedulerSession
    history_cache: HistoryTokenIds = field(default_factory=HistoryTokenIds)
    num_new_tokens: int = 0
    sampling_param: SamplingParam = field(default_factory=SamplingParam)
    status: MessageStatus = MessageStatus.WAITING
    logical_blocks: LogicalTokenBlocks = field(
        default_factory=LogicalTokenBlocks)
    sender_id: int = -1
    req_id: int = -1
    adapter_name: str = None
    arrive_time: float = 0.0
    meta: Any = None
    return_logits: bool = False
    random_offsets: int = 0

    def __post_init__(self):
        """post init."""
        self._history_len: int = 0
        self._tokens_len: int = len(self.history_cache)

    @property
    def block_size(self) -> int:
        """block size."""
        return self.session.block_size

    @property
    def history_len(self) -> int:
        """get history length."""
        return self._history_len

    @property
    def session_id(self) -> int:
        """get session id."""
        return self.session.session_id

    @property
    def token_ids(self) -> int:
        """token ids."""
        start = self.history_len
        end = start + self._tokens_len
        return self.history_cache[start:end]

    @property
    def history_token_ids(self) -> int:
        """history ids."""
        return self.history_cache[:self.history_len]

    @property
    def full_token_ids(self) -> int:
        """full token ids."""
        return self.history_cache[:self.num_all_tokens()]

    @property
    def num_history_ids(self):
        """num history ids."""
        return self._history_len

    @property
    def num_token_ids(self):
        return self._tokens_len

    def num_all_tokens(self) -> int:
        """num all tokens."""
        return self.history_len + self._tokens_len

    def update_token_ids(self, token_ids: Tensor):
        """Update token ids, old token ids will be added to history."""
        self._history_len += self._tokens_len
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.numpy()
        elif not isinstance(token_ids, np.ndarray):
            token_ids = np.array(token_ids)
        if token_ids.ndim == 0:
            token_ids = token_ids[None]
        self._tokens_len = len(token_ids)
        self.history_cache.append(token_ids)
        self.random_offsets += 1
        self.arrive_time = time.time()

    def set_step(self, step: int):
        """set step."""
        redo_size = self.history_len - step
        assert redo_size >= 0
        self._history_len -= redo_size
        self._tokens_len += redo_size
