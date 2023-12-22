# Copyright (c) OpenMMLab. All rights reserved.

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Union

import torch
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.distributed._tensor import DeviceMesh, Replicate, distribute_tensor
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, cached_file

from lmdeploy.pytorch.accel import LoadNoInit
from lmdeploy.utils import get_logger

from ..adapter.adapter import (AdapterWeightMap, get_indexed_lora_linears,
                               update_lora_linears)
from ..config import CacheConfig, ModelConfig
from ..models import patch
from ..utils import get_gpu_memory
from .cache_engine import CacheEngine

logger = get_logger('lmdeploy')

_PATCH_ARG_NAMES = ['context', 'use_origin']


def _update_cache_config(model_config: ModelConfig,
                         cache_config: CacheConfig,
                         gpu_id: int = 0):
    """Update the gpu mem and cpu mem according to model info.

    Args:
        model_config (ModelConfig): The config of the model.
        cache_config (CacheConfig): The config of the cache info.
        gpu_id (int): The GPU id to use.
    """
    GPU_MEM_PERCENT = 0.7
    SWAP_SPACE = 8 * (1 << 30)
    gpu_mem_physical_free, _ = get_gpu_memory(gpu_id)
    gpu_mem = gpu_mem_physical_free * GPU_MEM_PERCENT
    cpu_mem = SWAP_SPACE
    cache_block_size = CacheEngine.get_cache_block_size(
        cache_config.block_size, model_config)
    if cache_config.num_cpu_blocks == 0:
        cache_config.num_cpu_blocks = int(cpu_mem / cache_block_size)
    if cache_config.num_gpu_blocks == 0:
        cache_config.num_gpu_blocks = int(gpu_mem / cache_block_size)

    logger.info('block num: {}'.format(cache_config.num_gpu_blocks))


def _get_torch_dtype(config: Any, default: str = 'float16'):
    """Get the torch dtype from the model config.

    Args:
        config: Config of the hf model.
        default (str): default device type.
    """
    torch_dtype = getattr(config, 'torch_dtype', default)
    return eval(f'torch.{torch_dtype}')


@dataclass
class ModelInputs:
    """Input of the model."""
    input_ids: torch.LongTensor
    seq_length: torch.LongTensor
    attention_mask: torch.Tensor
    block_offsets: List[List[int]]
    position_ids: torch.LongTensor
    q_start_loc: torch.LongTensor
    history_lengths: List[int]
    is_decoding: bool
    local_adapter_ids: torch.LongTensor
    global_adapter_ids: torch.LongTensor
    adapter_offsets: torch.LongTensor
    max_rank: int
    meta: Any

    def to_device(self, device: str):
        """to device."""
        input_dict = asdict(self)
        out_dict = dict()
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            out_dict[k] = v

        return ModelInputs(**out_dict)


@dataclass
class StepContext:
    """context of Model.

    patched model might need extra information to perform inference. This
    dataclass provide these infos and tools.
    """
    inputs: ModelInputs
    block_offsets: torch.LongTensor
    position_ids: torch.LongTensor
    position_ids_1d: torch.LongTensor
    q_start_loc: torch.LongTensor
    history_lengths: torch.LongTensor
    seq_length: torch.LongTensor
    max_seq_length: int
    kv_seq_length: torch.LongTensor
    kv_caches: List
    is_decoding: bool
    world_size: int = 1
    json_config: Dict = None
    local_adapter_ids: torch.LongTensor = None
    global_adapter_ids: torch.LongTensor = None
    adapter_offsets: torch.LongTensor = None
    max_rank: int = 0

    _outputs: Dict = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        inputs: ModelInputs,
        world_size: int = 1,
        device: str = 'cuda',
        json_config: dict = None,
        kv_caches: List = None,
    ):
        """build step context.

        Args:
            inputs (ModelInputs): packaged model inputs.
            world_size (int): The distribution world size.
            device (str): The device of the tensors.
        """

        position_ids = inputs.position_ids
        max_seq_length = position_ids.size(-1)

        # seq_len + history_length
        kv_seq_length = position_ids[..., -1] + 1

        # position ids 1d
        seq_length = inputs.seq_length
        position_ids_1d = cls.get_position_ids_1d(position_ids, seq_length,
                                                  device)

        ret = StepContext(inputs=inputs,
                          block_offsets=inputs.block_offsets,
                          position_ids=inputs.position_ids,
                          position_ids_1d=position_ids_1d,
                          q_start_loc=inputs.q_start_loc,
                          history_lengths=inputs.history_lengths,
                          seq_length=inputs.seq_length,
                          max_seq_length=max_seq_length,
                          kv_seq_length=kv_seq_length,
                          kv_caches=kv_caches,
                          is_decoding=inputs.is_decoding,
                          world_size=world_size,
                          json_config=json_config,
                          local_adapter_ids=inputs.local_adapter_ids,
                          global_adapter_ids=inputs.global_adapter_ids,
                          adapter_offsets=inputs.adapter_offsets,
                          max_rank=inputs.max_rank)
        return ret

    @classmethod
    def tensorlize_block_offsets(cls, block_offsets, device):
        """tensorlize block_offsets."""
        import numpy as np
        offset_len = [len(offset) for offset in block_offsets]
        max_offsets_len = max(offset_len)
        batch_size = len(offset_len)
        pad_block_offsets = np.zeros((batch_size, max_offsets_len),
                                     dtype=np.int64)

        for pad_offset, offset, off_len in zip(pad_block_offsets,
                                               block_offsets, offset_len):
            pad_offset[:off_len] = offset

        block_offsets = torch.from_numpy(pad_block_offsets).to(device)
        return block_offsets

    @classmethod
    def get_position_ids_1d(cls,
                            position_ids: torch.LongTensor,
                            seq_length: torch.LongTensor,
                            device: str = 'cuda'):
        """get 1d position_ids."""
        if position_ids.size(1) == 1:
            position_ids_1d = position_ids.flatten()
        else:
            position_ids_1d = [
                ids[:l] for ids, l in zip(position_ids.cpu(), seq_length.cpu())
            ]
            position_ids_1d = torch.cat(position_ids_1d).to(device)
        return position_ids_1d

    def get_block_offsets(self):
        """return block offsets."""
        return self.block_offsets

    def set_output(self, key, value):
        """set output."""
        self._outputs[key] = value

    def get_output(self, key):
        """get output."""
        if key in self._outputs:
            return self._outputs[key]
        return None


def cache_swapping(cache_engine: CacheEngine, swap_in_map: dict,
                   swap_out_map: dict):
    """perform cache swapping."""
    issued_cache_op = False
    if len(swap_in_map) > 0:
        cache_engine.swap_in(swap_in_map)
        issued_cache_op = True
    if len(swap_out_map) > 0:
        cache_engine.swap_out(swap_out_map)
        issued_cache_op = True

    if issued_cache_op:
        cache_events = cache_engine.events
        for event in cache_events:
            event.wait()


def model_forward(
    patched_model: torch.nn.Module,
    inputs: ModelInputs,
    cache_engine: CacheEngine,
    json_config: dict = None,
    world_size: int = 1,
    stream: torch.cuda.Stream = None,
):
    """perform model forward."""
    stream = stream or torch.cuda.current_stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        # forward
        inputs = inputs.to_device('cuda')
        context = StepContext.new(
            inputs=inputs,
            world_size=world_size,
            json_config=json_config,
            kv_caches=cache_engine.gpu_cache,
        )
        output = patched_model.patched_forward(
            input_ids=inputs.input_ids,
            position_ids=inputs.position_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=cache_engine.gpu_cache,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            use_origin=False,
            context=context,
        )
    stream.synchronize()
    return dict(logits=output['logits'], custom_outputs=context._outputs)


def _load_adapters(hf_model: torch.nn.Module,
                   adapters: Dict[str, str],
                   device_map: str = 'cpu'):
    """load adapters."""
    if not adapters:
        return
    for name, path in adapters.items():
        logger.info(f'load adapter <{name}> from "{path}".')
        hf_model.load_adapter(path, name, device_map=device_map)


def _unparam_lora_weight(model: torch.nn.Module):
    """unparam lora weight.

    We don't want to move weight of lora to gpu.
    """
    from peft.tuners.lora import Linear as LoRALinear

    def _tensorize_weight(linear):
        """tensorize weight."""
        w = linear.weight
        del linear.weight
        linear.weight = w.data

    for _, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            lora_A = mod.lora_A
            lora_B = mod.lora_B
            for linear in lora_A.values():
                _tensorize_weight(linear)
            for linear in lora_B.values():
                _tensorize_weight(linear)


class BaseModelAgent:
    """Base model agent.

    load model on local gpu

    Args:
        model_path (str): The hugging face model path.
        model_config (ModelConfig): The config of the model.
        cache_config (CacheConfig): The config of the cache info.
        trust_remote_code (bool): Trust remote code
    """

    def __init__(self,
                 model_path: str,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 adapters: Dict[str, str] = None,
                 trust_remote_code: bool = True):
        self.model_config = model_config
        self.cache_config = cache_config
        torch_dtype = model_config.dtype

        self.patched_model = self._build_model(
            model_path,
            torch_dtype=torch_dtype,
            adapters=adapters,
            trust_remote_code=trust_remote_code)

        _update_cache_config(model_config, cache_config)

        self.cache_engine = CacheEngine(cache_config, model_config)
        self.stream = torch.cuda.Stream()

    def _build_model(self,
                     model_path: str,
                     torch_dtype: torch.dtype,
                     adapters: Dict[str, str] = None,
                     trust_remote_code: bool = True):
        """build patched model."""
        with LoadNoInit():
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code)
            hf_model.eval()
            hf_model.config.use_cache = True

        if adapters:
            _load_adapters(hf_model, adapters)

        patched_model = patch(hf_model, _PATCH_ARG_NAMES)

        if adapters:
            _unparam_lora_weight(patched_model)

        patched_model = patched_model.cuda()
        return patched_model

    def paging_adapters(self, weight_maps: List[AdapterWeightMap]):
        """load adapter."""
        lora_linears = get_indexed_lora_linears(self.patched_model)
        cpu_caches = self.cache_engine.cpu_cache
        num_blocks = self.cache_engine.num_cpu_blocks
        cpu_caches = [(kcache.view(num_blocks,
                                   -1), vcache.view(num_blocks, -1))
                      for kcache, vcache in cpu_caches]
        for weight_map in weight_maps:
            weight_map.cache_adapter(lora_linears, cpu_caches)
        update_lora_linears(lora_linears, weight_maps, device='cuda')

    def forward(self, inputs: ModelInputs, swap_in_map: Dict[int, int],
                swap_out_map: Dict[int, int]):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (Dict[int, int]): Cache maps to swap in.
            swap_out_map (Dict[int, int]): Cache maps to swap out.
        """

        cache_swapping(self.cache_engine,
                       swap_in_map=swap_in_map,
                       swap_out_map=swap_out_map)
        output = model_forward(
            self.patched_model,
            inputs,
            self.cache_engine,
            self.model_config.json_config,
            world_size=1,
            stream=self.stream,
        )
        return output


def _get_checkpoints(model_path: str):
    """get checkpoints."""
    try:
        torch_model_json_path = cached_file(model_path, WEIGHTS_INDEX_NAME)
        with open(torch_model_json_path, mode='r') as f:
            torch_model_json = json.load(f)

        weight_map = torch_model_json['weight_map']

        checkpoints = list(set(weight_map.values()))
        checkpoints = [cached_file(model_path, ckpt) for ckpt in checkpoints]
    except Exception:
        logger.warning(f'load failed, try load from {WEIGHTS_NAME}.')
        checkpoints = [cached_file(model_path, WEIGHTS_NAME)]

    return checkpoints


@dataclass
class TPResponse:
    ret_code: int
    error: Union[Exception, List[Exception]] = None
    data: Any = None


def _tp_build_model(
    rank: int,
    model_path: str,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    adapters: Dict[str, str],
    out_que: mp.Queue,
    world_size: int,
    trust_remote_code=True,
):
    """build tensor parallel model."""
    from accelerate import init_empty_weights

    error_code = 0
    error_type = None
    patched_model = None
    cache_engine = None

    def _broadcast_config(cache_config):
        """broadcast cache config, use minimum cache."""
        if rank == 0:
            gathered_configs = [None] * world_size
            dist.gather_object(cache_config, gathered_configs)
            num_gpu_blocks_list = [
                config.num_gpu_blocks for config in gathered_configs
            ]
            num_cpu_blocks_list = [
                config.num_cpu_blocks for config in gathered_configs
            ]
            min_num_gpu_blocks = min(num_gpu_blocks_list)
            min_num_cpu_blocks = min(num_cpu_blocks_list)
            cache_config.num_cpu_blocks = min_num_cpu_blocks
            cache_config.num_gpu_blocks = min_num_gpu_blocks
            config_list = [cache_config]
        else:
            gathered_configs = None
            dist.gather_object(cache_config, gathered_configs)
            config_list = [None]
        dist.broadcast_object_list(config_list)
        return config_list[0]

    try:
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)
        torch_dtype = _get_torch_dtype(config)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code)
        model.eval()
        model.config.use_cache = True

        if rank == 0:
            with LoadNoInit():
                param_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map='auto',
                    trust_remote_code=trust_remote_code)
                model.load_state_dict(param_model.state_dict(), assign=True)
                del param_model

        if adapters:
            device_map = 'auto'
            _load_adapters(model, adapters, device_map=device_map)

        patched_model = patch(
            model,
            extra_args=_PATCH_ARG_NAMES,
            rank=rank,
            world_size=world_size,
        )

        _update_cache_config(model_config, cache_config)
        cache_config = _broadcast_config(cache_config)
        cache_engine = CacheEngine(cache_config,
                                   model_config,
                                   rank=rank,
                                   world_size=world_size)
    except Exception as e:
        error_code = 1
        error_type = e

    # response
    error_code = torch.tensor(error_code).cuda(rank)
    dist.all_reduce(error_code)
    error_code = error_code.item()
    if error_code > 0:
        all_errors = [None] * world_size
        dist.all_gather_object(all_errors, error_type)
        if rank == 0:
            out_que.put(TPResponse(1, all_errors, cache_config))
        return
    else:
        if rank == 0:
            out_que.put(TPResponse(0, None, cache_config))

    return patched_model, cache_engine


def _tp_get_input(rank: int, in_que: mp.Queue, world_size: int):
    """get input tensor parallel."""
    device_mesh = DeviceMesh('cuda', list(range(world_size)))

    # broadcast meta info
    if rank == 0:
        inputs, swap_in_map, swap_out_map = in_que.get()
        inputs = asdict(inputs)
        input_tensors = dict(
            (k, v) for k, v in inputs.items() if isinstance(v, torch.Tensor))
        tensor_metas = dict(
            (name, (t.shape, t.dtype)) for name, t in input_tensors.items())
        other_metas = dict((k, v) for k, v in inputs.items()
                           if not isinstance(v, torch.Tensor))
        input_metas = (tensor_metas, other_metas)
        objs = [input_metas, swap_in_map, swap_out_map]
    else:
        objs = [None, None, None]

    dist.broadcast_object_list(objs)

    if rank != 0:
        input_metas = objs[0]
        tensor_metas, other_metas = input_metas
        input_tensors = dict((name, torch.empty(meta[0], dtype=meta[1]))
                             for name, meta in tensor_metas.items())

    updated_inputs = dict()
    for name, t in input_tensors.items():
        updated_inputs[name] = distribute_tensor(t,
                                                 device_mesh=device_mesh,
                                                 placements=[Replicate()
                                                             ]).to_local()

    inputs = updated_inputs
    inputs.update(other_metas)
    inputs = ModelInputs(**inputs)

    swap_in_map = objs[1]
    swap_out_map = objs[2]
    return inputs, swap_in_map, swap_out_map


def _tp_model_loop(
    rank: int,
    model_path: str,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    adapters: Dict[str, str],
    in_que: mp.Queue,
    out_que: mp.Queue,
    world_size: int,
    trust_remote_code=True,
):
    """Start model loops for tensor parallel model inference.

    Args:
        rank (int): Distribution rank.
        model_path (int): Path of the hugging face model. Could be
            local or online.
        model_config (ModelConfig): The config of the model.
        cache_config (CacheConfig): The config of the cache.
        in_que (mp.Queue): Input queue. Used to receive model input.
        out_que (mp.Queue): Output queue. Used to send the model output.
        world_size (int): The distribution world size.
    """
    stream = torch.cuda.Stream()
    patched_model, cache_engine = _tp_build_model(
        rank,
        model_path,
        model_config,
        cache_config,
        adapters,
        out_que=out_que,
        world_size=world_size,
        trust_remote_code=trust_remote_code)

    while True:
        inputs, swap_in_map, swap_out_map = _tp_get_input(
            rank, in_que, world_size)

        cache_swapping(cache_engine,
                       swap_in_map=swap_in_map,
                       swap_out_map=swap_out_map)

        output = model_forward(
            patched_model,
            inputs,
            cache_engine,
            model_config.json_config,
            world_size=world_size,
            stream=stream,
        )
        if rank == 0:
            resp_output = output
            out_que.put(TPResponse(0, None, resp_output))


def _start_tp_process(rank: int,
                      world_size: int,
                      func: Callable,
                      args: List = None,
                      kwargs: Dict = None):
    """Start the tensor parallel process.

    Args:
        rank (int): The distribution rank.
        world_size (int): The distribution world size.
        func (Callable): The function to be called in the process.
        args (List): The arguments of the func.
        kwargs (Dict): The keyword arguments of the func.
    """
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        with torch.cuda.device(rank), torch.no_grad():
            args = args or tuple()
            kwargs = kwargs or dict()
            func(rank, *args, **kwargs)
    except Exception as e:
        from traceback import print_exc

        logger.error(f'rank[{rank}]: {e}')
        print_exc()
        raise e


class TPModelAgent:
    """Tensor Parallelism model agent.

    load model on multiple GPUs

    Args:
        model_path (str): The hugging face model path.
        model_config (ModelConfig): The config of the model.
        cache_config (CacheConfig): The config of the cache info.
        trust_remote_code (bool): Trust remote code
    """

    def __init__(self,
                 model_path: str,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 world_size: int,
                 adapters: Dict[str, str] = None,
                 trust_remote_code: bool = True) -> None:
        mp.set_start_method('spawn')
        self.world_size = world_size
        self.model_config = model_config
        self.cache_config = cache_config
        self.tp_model_in_que = mp.Queue(10)
        self.tp_model_out_que = mp.Queue(10)

        self.patch_model_tp(model_path,
                            model_config=model_config,
                            cache_config=cache_config,
                            adapters=adapters,
                            in_que=self.tp_model_in_que,
                            out_que=self.tp_model_out_que,
                            world_size=world_size,
                            trust_remote_code=trust_remote_code)

    def patch_model_tp(self, model_path: str, model_config: ModelConfig,
                       cache_config: CacheConfig, adapters: Dict[str, str],
                       in_que: mp.Queue, out_que: mp.Queue, world_size: int,
                       trust_remote_code: bool):
        """Start tensor parallel sub process.

        Args:
            model_path (int): Path of the hugging face model.
                Could be local or online.
            extra_args (List[str]): The extra arguments to add to the
                patched model.
            model_config (ModelConfig): The config of the model.
            cache_config (CacheConfig): The config of the cache.
            in_que (mp.Queue): Input queue. Used to receive model input.
            out_que (mp.Queue): Output queue. Used to send the model output.
            world_size (int): The distribution world size.
        """
        self.mp_context = mp.spawn(
            _start_tp_process,
            args=(
                world_size,
                _tp_model_loop,
                (model_path, ),
                dict(model_config=model_config,
                     cache_config=cache_config,
                     adapters=adapters,
                     in_que=in_que,
                     out_que=out_que,
                     world_size=world_size,
                     trust_remote_code=trust_remote_code),
            ),
            nprocs=world_size,
            join=False,
            daemon=True,
        )
        resp: TPResponse = out_que.get()
        if resp.ret_code != 0:
            logger.error(f'Init tp model failed with error: {resp.error}')
            raise next(err for err in resp.error if err is not None)
        self.cache_config = resp.data

    def paging_adapters(self, weight_maps: List[AdapterWeightMap]):
        """load adapter."""
        raise NotImplementedError('TP lora is not supported for now.')

    def forward(self, inputs: Dict, swap_in_map: Dict[int, int],
                swap_out_map: Dict[int, int]):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (Dict[int, int]): Cache maps to swap in.
            swap_out_map (Dict[int, int]): Cache maps to swap out.
        """
        with torch.no_grad():
            self.tp_model_in_que.put((inputs, swap_in_map, swap_out_map))

        resp: TPResponse = self.tp_model_out_que.get()
        if resp.ret_code != 0:
            raise RuntimeError('tp forward failed.')

        return resp.data