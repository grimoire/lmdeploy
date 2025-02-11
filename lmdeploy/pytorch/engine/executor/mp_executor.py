# Copyright (c) OpenMMLab. All rights reserved.
# modify from vLLM: https://github.com/vllm-project/vllm/blob/main/vllm/v1/executor/multiproc_executor.py
import asyncio
import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory
import os
import pickle
import struct
from contextlib import contextmanager
from datetime import timedelta
from multiprocessing.context import SpawnContext
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.distributed import DistContext, get_dist_manager
from lmdeploy.pytorch.engine.model_agent import BaseModelAgent, build_model_agent
from lmdeploy.utils import get_logger

from .base import ExecutorBase

logger = get_logger('lmdeploy')

# 1m shared memory
SHARED_SIZE = 1 << 20
# data size
HEAD_SIZE = 8


def get_num_packages(data_size):
    """get num packages."""
    return (data_size + SHARED_SIZE - 1) // SHARED_SIZE


class SharedBuffer:
    """shared buffer."""

    def __init__(self, proc_id: int, notifier: Any, name: str = None):
        self.proc_id = proc_id
        self.notifier = notifier
        self.is_create = name is None
        if self.is_create:
            # double buffer
            self.shm = shared_memory.SharedMemory(create=True, size=(SHARED_SIZE + HEAD_SIZE) * 2)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self._buf_id = 0

        if proc_id >= 0:
            self.proc_mask = 1 << proc_id
        else:
            self.proc_mask = 0

    @contextmanager
    def acquire_buf(self):
        buf = self.shm.buf
        if self._buf_id == 0:
            out_buf = buf[:SHARED_SIZE + HEAD_SIZE]
        else:
            out_buf = buf[SHARED_SIZE + HEAD_SIZE:]
        yield out_buf

        self._buf_id = 1 - self._buf_id

    def name(self):
        return self.shm.name

    def pack_data(self, data):
        """pack data."""
        dumped_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        data_size = len(dumped_data)

        num_packs = get_num_packages(data_size)
        head = struct.pack('II', data_size, 0xff)

        for _ in range(num_packs):
            with self.acquire_buf() as buf:
                pac_size = min(len(dumped_data), SHARED_SIZE)
                packed_data = head + dumped_data[:pac_size]
                buf[:HEAD_SIZE + pac_size] = packed_data
                dumped_data = dumped_data[pac_size:]
                yield buf

    def send_all(self, data):
        """pack data."""
        for _ in self.pack_data(data):
            self.notifier.wait()

    async def send_all_async(self, data):
        """async pack data."""
        event_loop = asyncio.get_event_loop()
        for _ in self.pack_data(data):
            await event_loop.run_in_executor(None, self.notifier.wait)

    def _receive(self):
        """unpack data."""
        with self.acquire_buf() as buf:
            head = buf[:HEAD_SIZE]
            data_size, receiver_mask = struct.unpack('II', head)
            is_receiver = ((receiver_mask & self.proc_mask) > 0)

            pac_size = min(data_size, SHARED_SIZE)
            remain_size = data_size - pac_size

            dumped_data = b''
            if is_receiver:
                dumped_data += buf[HEAD_SIZE:HEAD_SIZE + pac_size]

        while remain_size > 0:
            self.notifier.wait()
            with self.acquire_buf() as buf:
                pac_size = min(remain_size, SHARED_SIZE)
                remain_size -= pac_size
                if not is_receiver:
                    continue
                dumped_data += buf[HEAD_SIZE:HEAD_SIZE + pac_size]

        data = pickle.loads(dumped_data)
        return data

    def receive(self):
        """unpack data."""
        self.notifier.wait()
        return self._receive()

    async def receive_async(self):
        """async receive data."""
        event_loop = asyncio.get_event_loop()
        await event_loop.run_in_executor(None, self.notifier.wait)
        return self._receive()

    def close(self):
        self.shm.close()
        if self.is_create:
            self.shm.unlink()


class MPExecutor(ExecutorBase):
    """Single node multi device Executor powered by multiprocess."""

    @staticmethod
    def _find_available_port() -> bool:
        """find available port."""
        import socket
        port = 29500
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
                port += 1

    @classmethod
    def setup_master_addr(cls):
        """setup master addr."""
        port = cls._find_available_port()
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', str(port))
        addr = os.environ['MASTER_ADDR']
        port = os.environ['MASTER_PORT']
        logger.info(f'MASTER_ADDR={addr}, MASTER_PORT={port}')

    def __init__(self,
                 model_path: str,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 backend_config: BackendConfig,
                 tokenizer: Any,
                 dp: int,
                 tp: int,
                 adapters: Dict[str, str] = None,
                 device_type: str = 'cuda'):
        """initialize Executor."""
        super().__init__(model_path=model_path,
                         model_config=model_config,
                         cache_config=cache_config,
                         backend_config=backend_config,
                         tokenizer=tokenizer,
                         dp=dp,
                         tp=tp,
                         adapters=adapters,
                         device_type=device_type)
        self.setup_master_addr()

        self.world_size = tp * dp
        mp_ctx = mp.get_context('spawn')
        self.comm_notifier = mp_ctx.Barrier(1 + self.world_size)
        self.comm_buf = SharedBuffer(-1, notifier=self.comm_notifier)
        self.comm_buf_name = self.comm_buf.name()

        self.procs: List[ExecutorProc] = []
        self.ret_bufs: List[SharedBuffer] = []
        for proc_id in range(self.world_size):
            proc = ExecutorProc(proc_id=proc_id, mp_ctx=mp_ctx)

            ret_notifier = mp_ctx.Barrier(2)
            ret_buf = SharedBuffer(0, notifier=ret_notifier)
            self.ret_bufs.append(ret_buf)
            proc.start(proc_id=proc_id,
                       comm_notifier=self.comm_notifier,
                       comm_buf_name=self.comm_buf_name,
                       ret_notifier=ret_notifier,
                       ret_buf_name=ret_buf.name(),
                       model_path=model_path,
                       model_config=model_config,
                       cache_config=cache_config,
                       backend_config=backend_config,
                       tokenizer=tokenizer,
                       dp=dp,
                       tp=tp,
                       adapters=adapters,
                       device_type=device_type)
            self.procs.append(proc)

        self.ret_all_mask = (1 << self.world_size) - 1

    def collective_rpc(self,
                       method: str,
                       args: Tuple[Any] = None,
                       kwargs: Dict[str, Any] = None,
                       call_async: bool = False,
                       return_mask: int = 0):
        """collective rpc."""
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        self.comm_buf.send_all(
            dict(
                method=method,
                args=args,
                kwargs=kwargs,
                call_async=call_async,
                return_mask=return_mask,
            ))

        if return_mask:
            outputs = [None] * len(self.ret_bufs)
            for proc_id, ret_buf in enumerate(self.ret_bufs):
                if bool(return_mask & (1 << proc_id)):
                    outputs[proc_id] = ret_buf.receive()
            return outputs

    async def collective_rpc_async(self,
                                   method: str,
                                   args: Tuple[Any] = None,
                                   kwargs: Dict[str, Any] = None,
                                   call_async: bool = False,
                                   return_mask: int = 0):
        """collective rpc."""
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        await self.comm_buf.send_all_async(
            dict(
                method=method,
                args=args,
                kwargs=kwargs,
                call_async=call_async,
                return_mask=return_mask,
            ))

        if return_mask:
            outputs = [None] * len(self.ret_bufs)
            for proc_id, ret_buf in enumerate(self.ret_bufs):
                if bool(return_mask & (1 << proc_id)):
                    outputs[proc_id] = await ret_buf.receive_async()
            return outputs

    def download_models(self):
        """download model."""
        raise NotImplementedError('Not Implemented.')

    def build_model(self):
        """build model."""
        self.collective_rpc('build_model')

    def gather_free_mem(self):
        """gather available memory."""
        return self.collective_rpc('get_free_mem', return_mask=self.ret_all_mask)

    def set_cache_config(self, cache_config: CacheConfig):
        """set all cache config."""
        self.collective_rpc('set_cache_config', args=(cache_config, ))

    def set_model_config(self, model_config: ModelConfig):
        """set all cache config."""
        self.collective_rpc('set_model_config', args=(model_config, ))

    def build_graph_runner(self):
        """build graph runner."""
        self.collective_rpc('build_graph_runner')

    def build_cache_engine(self):
        """build cache engine."""
        self.collective_rpc('build_cache_engine')

    def start(self, forward_event: asyncio.Event):
        """start engine loop."""
        forward_event.clear()
        self.collective_rpc('start')
        forward_event.set()

    async def forward_async(self, inputs):
        """start forward."""
        await self.collective_rpc_async('set_forward_inputs', args=(inputs, ))

    async def get_output_async(self):
        """get output async."""
        return (await self.collective_rpc_async('get_output_async', call_async=True, return_mask=1))[0]

    def get_input_processor(self):
        """get input processor."""
        self.collective_rpc('get_input_processor', return_mask=1)

    def stop(self):
        """stop engine loop."""

    def release(self):
        """release."""
        for proc in self.procs:
            proc.close()

        self.comm_buf.close()
        for ret_buf in self.ret_bufs:
            ret_buf.close()


def init_dist_environ(rank: int, world_size: int, nproc_per_node: int):
    """init environ."""
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank % nproc_per_node)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(nproc_per_node)


class ExecutorProc:

    def __init__(self, proc_id: int, mp_ctx: SpawnContext):
        """executor proc."""
        self.proc_id = proc_id
        self.mp_ctx = mp_ctx
        self._proc = None

    def start(self, **kwargs):
        """start proc."""
        proc = self.mp_ctx.Process(target=self._main_loop, kwargs=kwargs, daemon=True)
        proc.start()
        self._proc = proc

    def close(self):
        """stop proc."""
        if self._proc is None:
            return
        if not self._proc.is_alive():
            return
        self._proc.close()

    @staticmethod
    def init_process_group(rank: int, world_size: int, nproc_per_node: int):
        """init process group."""
        DIST_TIMEOUT = timedelta(days=35600)
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size, timeout=DIST_TIMEOUT)
        assert dist.is_initialized()
        init_dist_environ(rank, world_size, nproc_per_node)

    def _main_loop(
        self,
        proc_id: int,
        comm_notifier: Any,
        comm_buf_name: str,
        ret_notifier: Any,
        ret_buf_name: str,
        model_path: str,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        tokenizer: Any,
        dp: int,
        tp: int,
        adapters: Dict[str, str] = None,
        device_type: str = 'cuda',
    ):
        """main loop."""

        world_size = tp * dp
        self.init_process_group(proc_id, world_size, 1)

        dist_ctx = DistContext.build(proc_id, tp, dp)
        device_ctx = DeviceContext(device_type=device_type)

        dist_mgr = get_dist_manager()
        device_mgr = get_device_manager()

        torch.cuda.set_device(proc_id)
        with dist_mgr.context(dist_ctx), device_mgr.context(device_ctx):
            model_agent = build_model_agent(model_path=model_path,
                                            model_config=model_config,
                                            cache_config=cache_config,
                                            backend_config=backend_config,
                                            tokenizer=tokenizer,
                                            adapters=adapters)

            comm_buf = SharedBuffer(proc_id, notifier=comm_notifier, name=comm_buf_name)
            ret_buf = SharedBuffer(-1, notifier=ret_notifier, name=ret_buf_name)
            try:
                event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(event_loop)

                event_loop.run_until_complete(
                    self._main_loop_impl(proc_id, comm_buf=comm_buf, ret_buf=ret_buf, model_agent=model_agent))
            finally:
                comm_buf.close()

    async def _main_loop_impl(self, proc_id: int, comm_buf: SharedBuffer, ret_buf: SharedBuffer,
                              model_agent: BaseModelAgent):
        """main loop."""
        proc_mask = 1 << proc_id
        while True:
            command = await comm_buf.receive_async()
            method = command['method']
            args = command.get('args', list())
            kwargs = command.get('kwargs', dict())
            call_async = command.get('call_async', False)
            return_mask = command.get('return_mask', True)
            need_return = bool(proc_mask & return_mask)

            func = getattr(model_agent, method, None)
            assert func is not None

            if call_async:
                ret = await func(*args, **kwargs)
            else:
                ret = func(*args, **kwargs)

            if need_return:
                await ret_buf.send_all_async(ret)
