# Copyright (c) OpenMMLab. All rights reserved.
from .base_request_manager import Request, RequestType, Response

__all__ = ['Request', 'RequestType', 'Response']


def build_request_manager(manager_type: str = 'default'):
    """build request manager."""
    if manager_type == 'coro':
        from .coro_request_manager import CoroRequestManager
        return CoroRequestManager()
    elif manager_type == 'default':
        from .default_request_manager import RequestManager
        return RequestManager()
    else:
        raise TypeError(f'Unknown request manager type: {manager_type}')
