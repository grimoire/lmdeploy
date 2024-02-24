# Copyright (c) OpenMMLab. All rights reserved.
import enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from lmdeploy.messages import ResponseType
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class RequestType(enum.Enum):
    """Request type."""

    ADD_SESSION = enum.auto()
    ADD_MESSAGE = enum.auto()
    STOP_SESSION = enum.auto()
    END_SESSION = enum.auto()
    STOP_ENGINE = enum.auto()
    RESUME_ENGINE = enum.auto()


@dataclass
class Request:
    """Request."""

    type: RequestType
    sender_id: int
    req_id: int
    data: Any = None


ReqList = List[Request]


@dataclass
class Response:
    """Response."""

    type: ResponseType
    sender_id: int
    req_id: int
    data: Any = None
    err_msg: str = ''


@dataclass
class BaseRequestSender:
    """Request sender.

    Args:
        sender_id (int): The id of the sender
    """

    sender_id: int

    @classmethod
    def new(cls, sender_id: int, *args, **kwargs):
        """new sender."""
        raise NotImplementedError('Not implemented')

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
        raise NotImplementedError('Not implemented')

    def send_async(self, req_type: RequestType, data: Any) -> int:
        """send request asynchronize."""
        raise NotImplementedError('Not implemented')

    def recv_any(self, que_timeout: float = None) -> Response:
        """receive any response."""
        raise NotImplementedError('Not implemented')

    def recv_all(self, req_id: int, block: bool = True):
        """revceive all response with req_id."""
        raise NotImplementedError('Not implemented')

    def recv(self, req_id: int, que_timeout: float = None) -> Response:
        """receive response of given request id."""
        raise NotImplementedError('Not implemented')

    async def async_recv(self,
                         req_id: int,
                         que_timeout: float = None) -> Response:
        """receive response of given request id async."""
        raise NotImplementedError('Not implemented')

    async def async_send(self,
                         req_type: RequestType,
                         data: Any,
                         que_timeout: float = None):
        """send and receive synchronize."""
        raise NotImplementedError('Not implemented')

    def send(self,
             req_type: RequestType,
             data: Any,
             que_timeout: float = None) -> Response:
        """send and receive synchronize."""
        raise NotImplementedError('Not implemented')

    def response_callback(self, resp: Response, timeout: float = None):
        """response callback."""
        raise NotImplementedError('Not implemented')

    def is_loop_alive(self):
        """check if main loop is alive."""
        raise NotImplementedError('Not implemented')


class BaseRequestManager:
    """Request manager."""

    def __init__(self):
        self.senders: Dict[int, BaseRequestSender] = dict()
        self.callbacks: Dict[RequestType, Callable] = dict()
        self.request_priority: List[RequestType] = [
            RequestType.STOP_ENGINE, RequestType.STOP_SESSION,
            RequestType.END_SESSION, RequestType.ADD_SESSION,
            RequestType.ADD_MESSAGE
        ]

    def start_loop(self, loop: Callable):
        """start main loop."""
        raise NotImplementedError('Not implemented')

    def is_loop_alive(self):
        """check if main loop is alive."""
        raise NotImplementedError('Not implemented')

    def build_sender(self):
        """create a new sender."""
        raise NotImplementedError('Not implemented')

    def has_requests(self):
        """has unprocessed request."""
        raise NotImplementedError('Not implemented')

    def get_all_requests(self) -> Dict[RequestType, Request]:
        """get all requests in current queue."""
        raise NotImplementedError('Not implemented')

    def bind_func(self, req_type: RequestType, callback: Callable):
        """bind handler for given request type."""
        self.callbacks[req_type] = callback

    def set_request_priority(self, priority: List[RequestType]):
        """set the priority of request type."""
        self.request_priority = priority

    def response(self, resp: Response, timeout: float = None):
        """send response."""
        if resp.sender_id not in self.senders:
            logger.warning(f'sender {resp.sender_id} not exist. '
                           f'Send {resp} failed.')
            return
        self.senders[resp.sender_id].response_callback(resp, timeout=timeout)

    def process_request(self, req_type: RequestType, reqs: ReqList, **kwargs):
        """process reqs with given req type."""
        # get callback
        func = self.callbacks.get(req_type, None)
        if func is not None:
            func(reqs, **kwargs)
        else:
            # TODO: send error message
            for req in reqs:
                resp = Response(ResponseType.HANDLER_NOT_EXIST,
                                sender_id=req.sender_id,
                                req_id=req.req_id,
                                err_msg=(f'callback for {req_type}'
                                         ' not exists.'))
                self.response(resp)

    def step(self, **kwargs):
        """handle requests."""
        reqs_by_type = self.get_all_requests()

        # handle requests
        for req_type in self.request_priority:
            # request exists
            if req_type not in reqs_by_type or len(reqs_by_type) == 0:
                continue

            reqs: ReqList = reqs_by_type[req_type]
            self.process_request(req_type, reqs, **kwargs)
