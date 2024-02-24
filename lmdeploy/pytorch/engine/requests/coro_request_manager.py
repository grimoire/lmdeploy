# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from lmdeploy.utils import get_logger

from .base_request_manager import (BaseRequestManager, BaseRequestSender,
                                   ReqList, Request, RequestType, Response)

logger = get_logger('lmdeploy')


def is_in_running_loop():
    try:
        asyncio.get_running_loop()
        return True
    except Exception:
        return False


@dataclass
class CoroRequestSender(BaseRequestSender):
    """Coroutine Request sender.

    Args:
        sender_id (int): The id of the sender
    """

    sender_id: int
    manager: 'CoroRequestManager'
    resp_dict: Dict[int, List[Response]] = field(default_factory=dict)
    _next_req_id: int = 0
    _resp_que: asyncio.Queue = None

    @classmethod
    def new(cls, sender_id: int, manager: 'CoroRequestManager'):
        """new."""
        return cls(sender_id=sender_id, manager=manager)

    @property
    def resp_que(self):
        if self.manager._loop_task is None:
            self.manager.create_loop_task()
        if self._resp_que is None:
            self._resp_que = asyncio.Queue()
        return self._resp_que

    @property
    def req_que(self):
        return self.manager.requests

    def _push_resp(self, req_id: int, resp: Response):
        """push response."""
        self.resp_dict.setdefault(req_id, [])
        self.resp_dict[req_id].append(resp)

    def _pop_resp(self, req_id: int, default: Any = None):
        """pop response."""
        if req_id not in self.resp_dict:
            return default
        resps = self.resp_dict[req_id]
        ret = resps.pop(0)
        if len(resps) == 0:
            self.resp_dict.pop(req_id)
        return ret

    def _prefetch_resps(self):
        """prefetch from resp que."""
        num_resps = self.resp_que.qsize()
        for _ in range(num_resps):
            resp: Response = self.resp_que.get_nowait()
            req_id = resp.req_id
            self._push_resp(req_id, resp)

    def _gather_request(self, req_types: List[RequestType], data: List[Any]):
        """gather requests."""
        if self.manager._loop_task is None:
            self.manager.create_loop_task()
        if not self.is_loop_alive():
            logger.error('Engine main loop stopped.')
            exit(1)
        assert len(req_types) == len(data)
        batch_size = len(req_types)

        req_ids = list(range(self._next_req_id,
                             self._next_req_id + batch_size))
        self._next_req_id += batch_size

        reqs = [
            Request(type=rtype,
                    sender_id=self.sender_id,
                    req_id=req_id,
                    data=rdata)
            for req_id, rtype, rdata in zip(req_ids, req_types, data)
        ]
        return req_ids, reqs

    async def async_batched_send_async(self, req_types: List[RequestType],
                                       data: List[Any]):
        """Batched send request asynchronize."""
        req_ids, reqs = self._gather_request(req_types, data)
        await self.req_que.put(reqs)
        return req_ids

    async def async_send_async(self, req_type: RequestType, data: Any):
        """send request asynchronize."""
        return (await self.async_batched_send_async(req_types=[req_type],
                                                    data=[data]))[0]

    def batched_send_async(self, req_types: List[RequestType],
                           data: List[Any]) -> List[int]:
        """Batched send request asynchronize."""
        coro = self.async_batched_send_async(req_types, data)
        return asyncio.run(coro)

    def send_async(self, req_type: RequestType, data: Any) -> int:
        """send request asynchronize."""
        return self.batched_send_async(req_types=[req_type], data=[data])[0]

    def recv_any(self, que_timeout: float = None) -> Response:
        """receive any response."""
        self._prefetch_resps()
        for req_id in self.resp_dict:
            ret = self._pop_resp(req_id, default=None)
            if ret is not None:
                return ret

    def recv_all(self, req_id: int, block: bool = True):
        """revceive all response with req_id."""
        self._prefetch_resps()
        resps = self.resp_dict.pop(req_id, [])
        return resps

    async def async_recv(self,
                         req_id: int,
                         que_timeout: float = None) -> Response:
        """receive response of given request id async."""
        ret = self._pop_resp(req_id, default=None)
        if ret is not None:
            return ret

        # check resp que
        while True:
            resp: Response = await self.resp_que.get()
            if resp.req_id != req_id:
                self._push_resp(req_id, resp)
            else:
                return resp

    def recv(self, req_id: int, que_timeout: float = None) -> Response:
        """receive response of given request id."""
        coro = self.async_recv(req_id, que_timeout)
        return asyncio.run(coro)

    async def async_send(self,
                         req_type: RequestType,
                         data: Any,
                         que_timeout: float = None):
        """send and receive synchronize."""
        req_id = await self.async_send_async(req_type, data)
        return await self.async_recv(req_id, que_timeout=que_timeout)

    def send(self,
             req_type: RequestType,
             data: Any,
             que_timeout: float = None) -> Response:
        """send and receive synchronize."""
        req_id = self.send_async(req_type, data)
        return self.recv(req_id, que_timeout=que_timeout)

    def response_callback(self, resp: Response, timeout: float = None):
        """response callback."""
        self.resp_que.put_nowait(resp)

    def is_loop_alive(self):
        """is loop alive."""
        return self.manager.is_loop_alive()


def _raise_exception_on_finish(task: asyncio.Task) -> None:
    msg = ('Task finished unexpectedly. This should never happen! '
           ' See stack trace above for the actual cause.')
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as exc:
        raise RuntimeError(msg) from exc


class CoroRequestManager(BaseRequestManager):
    """Coroutine Request manager."""

    def __init__(self):
        super().__init__()
        self.requests: asyncio.Queue = None
        self._loop_task: asyncio.Future = None
        self._loop_coro: Callable = None
        self._next_sender_id = 0

    def create_loop_task(self):
        """create coro task."""
        loop_unshielded = asyncio.get_event_loop().create_task(
            self._loop_coro())
        loop_unshielded.add_done_callback(_raise_exception_on_finish)
        self._loop_task = asyncio.shield(loop_unshielded)
        self.requests = asyncio.Queue()
        return self._loop_task

    def start_loop(self, loop: asyncio.Task):
        """start main loop."""
        self._loop_coro = loop

    def is_loop_alive(self):
        """check if main loop is alive."""
        if self._loop_task is None:
            logger.warning('loop task has not been created.')
            return False
        if self._loop_task.get_loop() != asyncio.get_event_loop():
            logger.warning('Current event loop os is different from'
                           ' the one bound to loop task!')
            return False
        return not self._loop_task.done()

    def build_sender(self):
        """create a new sender."""
        sender_id = self._next_sender_id
        self._next_sender_id += 1
        new_sender = CoroRequestSender.new(sender_id, self)
        self.senders[sender_id] = new_sender
        return new_sender

    def has_requests(self):
        """has unprocessed request."""
        if self.requests is None:
            return False
        return not self.requests.empty()

    def get_all_requests(self) -> Dict[RequestType, Request]:
        """get all requests in current queue."""
        num_reqs = self.requests.qsize()
        reqs: ReqList = []
        for _ in range(num_reqs):
            elem = self.requests.get_nowait()
            if isinstance(elem, Request):
                elem = [elem]
            reqs += elem

        # gather requests
        reqs_by_type: Dict[RequestType, Request] = dict(
            (t, []) for t in RequestType)
        for req in reqs:
            reqs_by_type[req.type].append(req)
        return reqs_by_type
