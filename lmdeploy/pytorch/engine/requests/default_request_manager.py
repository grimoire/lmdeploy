# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Callable, ClassVar, Dict, List

from lmdeploy.utils import get_logger

from .base_request_manager import (BaseRequestManager, BaseRequestSender,
                                   ReqList, Request, RequestType, Response)

logger = get_logger('lmdeploy')


@dataclass
class RequestSender(BaseRequestSender):
    """Request sender.

    Args:
        sender_id (int): The id of the sender
    """

    sender_id: int
    req_que: Queue
    resp_que: Queue = field(default_factory=Queue)
    resp_dict: Dict[int, List[Response]] = field(default_factory=dict)
    THREAD_ALIVE_INTERVAL: ClassVar[float] = 1.0
    _next_req_id: int = 0
    _thread: Thread = None

    @classmethod
    def new(cls, sender_id: int, req_que: Queue, thread: Thread):
        """new sender."""
        return cls(sender_id=sender_id, req_que=req_que, _thread=thread)

    def _resp_que_get(self, block: bool = True, timeout: float = None):
        """warp of resp_que.get."""
        if not block:
            return self.resp_que.get(block=block, timeout=timeout)
        timeout_counter = timeout or float(1 << 30)
        while timeout_counter > self.THREAD_ALIVE_INTERVAL:
            try:
                return self.resp_que.get(timeout=self.THREAD_ALIVE_INTERVAL)
            except Empty:
                timeout_counter -= self.THREAD_ALIVE_INTERVAL
            if not self.is_loop_alive():
                logger.error('Engine main loop stopped.')
                exit(1)

        return self.resp_que.get(timeout=timeout_counter)

    async def _async_resp_que_get(self,
                                  block: bool = True,
                                  timeout: float = None):
        """warp of resp_que.get."""
        if not block:
            return self.resp_que.get(block=block, timeout=timeout)
        timeout_counter = timeout or float(1 << 30)
        while timeout_counter > self.THREAD_ALIVE_INTERVAL:
            if self.resp_que.qsize() == 0:
                await asyncio.sleep(self.THREAD_ALIVE_INTERVAL)
                timeout_counter -= self.THREAD_ALIVE_INTERVAL
            else:
                return self.resp_que.get(block=False)
            if not self.is_loop_alive():
                logger.error('Engine main loop stopped.')
                exit(1)

        await asyncio.sleep(self.THREAD_ALIVE_INTERVAL)
        return self.resp_que.get(block=False)

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

    def _prefetch_resps(self, block=True):
        """prefetch from resp que."""
        num_resps = self.resp_que.qsize()
        for _ in range(num_resps):
            resp: Response = self._resp_que_get(block=block)
            req_id = resp.req_id
            self._push_resp(req_id, resp)

    def _gather_request(self, req_types: List[RequestType], data: List[Any]):
        """gather requests."""
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
        raise NotImplementedError('Not implemented')

    async def async_send_async(self, req_type: RequestType, data: Any):
        """send request asynchronize."""
        raise NotImplementedError('Not implemented')

    def batched_send_async(self, req_types: List[RequestType],
                           data: List[Any]) -> List[int]:
        """Batched send request asynchronize."""
        req_ids, reqs = self._gather_request(req_types, data)
        self.req_que.put(reqs)
        return req_ids

    def send_async(self, req_type: RequestType, data: Any) -> int:
        """send request asynchronize."""
        return self.batched_send_async(req_types=[req_type], data=[data])[0]

    def recv_any(self, que_timeout: float = None) -> Response:
        """receive any response."""
        # check resp dict
        self._prefetch_resps()
        for req_id in self.resp_dict:
            ret = self._pop_resp(req_id, default=None)
            if ret is not None:
                return ret

        # check resp que
        return self._resp_que_get(timeout=que_timeout)

    def recv_all(self, req_id: int, block: bool = True):
        """revceive all response with req_id."""
        self._prefetch_resps(block)
        resps = self.resp_dict.pop(req_id, [])
        return resps

    def recv(self, req_id: int, que_timeout: float = None) -> Response:
        """receive response of given request id."""
        # check resp dict
        ret = self._pop_resp(req_id, default=None)
        if ret is not None:
            return ret

        # check resp que
        while True:
            resp: Response = self._resp_que_get(timeout=que_timeout)
            if resp.req_id != req_id:
                self._push_resp(req_id, resp)
            else:
                return resp

    async def async_recv(self,
                         req_id: int,
                         que_timeout: float = None) -> Response:
        """receive response of given request id async."""
        self._prefetch_resps(block=False)
        ret = self._pop_resp(req_id, default=None)
        if ret is not None:
            return ret

        # check resp que
        while True:
            resp: Response = await self._async_resp_que_get(timeout=que_timeout
                                                            )
            if resp.req_id != req_id:
                self._push_resp(req_id, resp)
            else:
                return resp

    def send(self,
             req_type: RequestType,
             data: Any,
             que_timeout: float = None) -> Response:
        """send and receive synchronize."""
        req_id = self.send_async(req_type, data)

        return self.recv(req_id, que_timeout=que_timeout)

    def is_loop_alive(self):
        """check if main loop is alive."""
        if self._thread is None:
            return False
        return self._thread.is_alive()

    def response_callback(self, resp: Response, timeout: float = None):
        """response callback."""
        self.resp_que.put(resp, timeout=timeout)


class RequestManager(BaseRequestManager):
    """Request manager."""

    def __init__(self):
        super().__init__()
        self._next_sender_id = 0
        self.requests = Queue()
        self.mutex = Lock()
        self._loop_thread = None

    def start_loop(self, loop: Callable):
        """start main loop."""
        loop_thread = Thread(target=loop, daemon=True)
        loop_thread.start()
        self._loop_thread = loop_thread
        return loop_thread

    def is_loop_alive(self):
        """check if main loop is alive."""
        if self._loop_thread is None:
            return False
        return self._loop_thread.is_alive()

    def build_sender(self):
        """create a new sender."""
        with self.mutex:
            assert self._loop_thread is not None
            sender_id = self._next_sender_id
            self._next_sender_id += 1
            new_sender = RequestSender.new(sender_id, self.requests,
                                           self._loop_thread)
            self.senders[sender_id] = new_sender
            return new_sender

    def has_requests(self):
        """has unprocessed request."""
        return not self.requests.empty()

    def get_all_requests(self) -> Dict[RequestType, Request]:
        """get all requests in current queue."""
        num_reqs = self.requests.qsize()
        reqs: ReqList = []
        tmp = num_reqs
        while tmp:
            tmp -= 1
            elem = self.requests.get(block=False)
            if isinstance(elem, Request):
                elem = [elem]
            reqs += elem

        # gather requests
        reqs_by_type: Dict[RequestType, Request] = dict(
            (t, []) for t in RequestType)
        for req in reqs:
            reqs_by_type[req.type].append(req)
        return reqs_by_type
