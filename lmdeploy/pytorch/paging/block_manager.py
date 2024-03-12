# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from ..config import CacheConfig


class LogicalMemory:
    """Logical memory blocks."""

    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks

        self.phy_map: np.ndarray = np.zeros(self._num_blocks, dtype=np.int64)

    def get_physical_blocks(self, logical_address: np.ndarray):
        """get physical address."""
        if isinstance(logical_address,
                      np.ndarray) and len(logical_address) == 0:
            return np.empty((0, ), dtype=np.int64)
        return self.phy_map[logical_address]

    @property
    def num_blocks(self):
        """get num blocks."""
        return self._num_blocks


class BlockStack:
    """block stack."""

    def __init__(self, num_blocks: int, offset: int = 0):
        self._offset = offset
        self._num_blocks = num_blocks
        self._free_blocks = np.arange(num_blocks, dtype=np.int64) + offset
        self._free_count = num_blocks

    @property
    def num_blocks(self):
        """num blocks."""
        return self._num_blocks

    @property
    def offset(self):
        """offset."""
        return self._offset

    def __len__(self):
        """free len."""
        return self._free_count

    def pop(self, num_blocks: int):
        """pop."""
        if num_blocks == 0:
            return np.empty((0, ), dtype=np.int64)
        num_freed = len(self)
        assert num_blocks <= num_freed, 'No free blocks available'

        blocks = self._free_blocks[num_freed - num_blocks:num_freed]
        self._free_count -= num_blocks
        return blocks

    def push(self, blocks: np.ndarray):
        """push."""
        num_new_blocks = len(blocks)
        if num_new_blocks == 0:
            return

        num_freed = len(self)
        assert num_freed + num_new_blocks <= self.num_blocks, (
            'Too much free blocks')
        self._free_blocks[num_freed:num_freed + num_new_blocks] = blocks
        self._free_count += num_new_blocks


class PhysicalAllocator:
    """The physical block allocator.

    The allocator won't allocate real memory. It is used to support block
    manager.
    """

    def __init__(self, num_blocks: int, offset: int = 0):
        self._stack = BlockStack(num_blocks, offset=offset)

    @property
    def num_blocks(self):
        """num blocks."""
        return self._stack.num_blocks

    @property
    def offset(self):
        """offset."""
        return self._stack.offset

    def allocate(self, num_blocks: int):
        """Allocate block from block pool."""
        return self._stack.pop(num_blocks)

    def free(self, blocks: np.ndarray):
        """Free block to block pool.

        We will not check if the block is valid
        """
        return self._stack.push(blocks)

    def get_num_free_blocks(self):
        """Get numbers of free blocks."""
        return len(self._stack)


class LogicalAllocator:
    """The logical block allocator."""

    def __init__(self, num_cpu_blocks: int, num_gpu_blocks: int) -> None:
        self._log_mem = LogicalMemory(num_cpu_blocks + num_gpu_blocks)

        self._cpu_mem_offset = num_gpu_blocks
        self._gpu_allocator = PhysicalAllocator(num_gpu_blocks, 0)
        self._cpu_allocator = PhysicalAllocator(num_cpu_blocks,
                                                self._cpu_mem_offset)

        self._stack = BlockStack(num_cpu_blocks + num_gpu_blocks)

    @property
    def num_blocks(self):
        """num blocks."""
        return self._stack.num_blocks

    def get_phy_allocator(self, device: str):
        """get allocator."""
        if device == 'gpu':
            return self._gpu_allocator
        elif device == 'cpu':
            return self._cpu_allocator
        else:
            raise ValueError(f'Unsupported device: {device}')

    def allocate(self, num_blocks: int, device: str = 'gpu'):
        """allocate logical blocks."""
        if num_blocks == 0:
            return np.empty((0, ), dtype=np.int64)
        phy_allocator = self.get_phy_allocator(device)
        blocks = self._stack.pop(num_blocks)
        phy_blocks = phy_allocator.allocate(num_blocks)
        self._log_mem.phy_map.put(blocks, phy_blocks)
        return blocks.copy()

    def free(self, blocks: np.ndarray, device: str = None):
        """Free logical block."""
        if len(blocks) == 0:
            return
        phy_blocks = self.get_physical_blocks(blocks)

        cpu_blocks = None
        gpu_blocks = None
        if device is None:
            cpu_blocks = phy_blocks[phy_blocks >= self._cpu_mem_offset]
            gpu_blocks = phy_blocks[phy_blocks < self._cpu_mem_offset]
        elif device == 'cpu':
            cpu_blocks = phy_blocks
        elif device == 'gpu':
            gpu_blocks = phy_blocks
        else:
            raise ValueError(f'Unknown device: {device}')

        if cpu_blocks is not None and len(cpu_blocks) > 0:
            self._cpu_allocator.free(cpu_blocks)
        if gpu_blocks is not None and len(gpu_blocks) > 0:
            self._gpu_allocator.free(gpu_blocks)

        self._stack.push(blocks)

    def get_num_free_blocks(self):
        """Get numbers of free blocks."""
        return len(self._stack)

    def get_physical_blocks(self, blocks: np.ndarray):
        """get physical address."""
        return self._log_mem.get_physical_blocks(blocks)

    def cpu_mem_offset(self):
        """get cpu mem offset in unified physical memory."""
        return self._cpu_mem_offset

    def count_cpu_blocks(self, blocks: np.ndarray):
        """count cpu blocks."""
        return len(blocks) - self.count_gpu_blocks(blocks)

    def count_gpu_blocks(self, blocks: np.ndarray):
        """count gpu blocks."""
        phy_blocks = self.get_physical_blocks(blocks)
        return np.count_nonzero(phy_blocks < self.cpu_mem_offset())

    def update_phy_map(self, log_blocks: np.ndarray, phy_blocks: np.ndarray):
        """update physical map."""
        assert len(phy_blocks) == len(log_blocks)
        self._log_mem.phy_map.put(log_blocks, phy_blocks)


BlockTable = np.ndarray


class BlockManager:

    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks

        self.allocator = LogicalAllocator(num_cpu_blocks, num_gpu_blocks)

    def can_allocate(self, num_blocks: int, device: str = 'gpu'):
        """can allocate."""
        if device == 'cpu':
            num_free = self.get_num_free_cpu_blocks()
        elif device == 'gpu':
            num_free = self.get_num_free_gpu_blocks()
        else:
            raise ValueError(f'Unknown device: {device}')

        return num_blocks <= num_free

    def allocate(self, num_blocks: int, device: str = 'gpu'):
        """allocate."""
        return self.allocator.allocate(num_blocks, device)

    def free(self, blocks: np.ndarray, device: str = None):
        """free."""
        return self.allocator.free(blocks, device)

    def can_swap_in(self, blocks: np.ndarray):
        """can swap in.

        do not check the device of the blocks
        """
        num_free = self.get_num_free_gpu_blocks()
        return len(blocks) < num_free

    def can_swap_out(self, blocks: np.ndarray):
        """can swap out.

        do not check the device of the blocks
        """
        num_free = self.get_num_free_cpu_blocks()
        return len(blocks) < num_free

    def _swap(self, blocks: np.ndarray, from_phy_allocator: PhysicalAllocator,
              to_phy_allocator: PhysicalAllocator):
        """swap."""
        num_blocks = len(blocks)
        if num_blocks == 0:
            return dict()

        num_blocks = len(blocks)
        phy_blocks = self.allocator.get_physical_blocks(blocks)
        new_phy_blocks = to_phy_allocator.allocate(num_blocks)

        from_phy_allocator.free(phy_blocks)
        self.allocator.update_phy_map(blocks, new_phy_blocks)

        return phy_blocks, new_phy_blocks

    def swap_in(self, blocks: np.ndarray):
        """swap in."""

        from_phy_allocator = self.allocator.get_phy_allocator('cpu')
        to_phy_allocator = self.allocator.get_phy_allocator('gpu')

        old_phy_blocks, new_phy_blocks = self._swap(blocks, from_phy_allocator,
                                                    to_phy_allocator)

        swap_map = dict(
            zip(old_phy_blocks - self.allocator.cpu_mem_offset(),
                new_phy_blocks))
        return swap_map

    def swap_out(self, blocks: np.ndarray):
        """swap out."""

        from_phy_allocator = self.allocator.get_phy_allocator('gpu')
        to_phy_allocator = self.allocator.get_phy_allocator('cpu')

        old_phy_blocks, new_phy_blocks = self._swap(blocks, from_phy_allocator,
                                                    to_phy_allocator)

        swap_map = dict(
            zip(old_phy_blocks,
                new_phy_blocks - self.allocator.cpu_mem_offset()))
        return swap_map

    def get_num_free_gpu_blocks(self) -> int:
        """Get number of free gpu blocks."""
        return self.allocator.get_phy_allocator('gpu').get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        """Get number of free cpu blocks."""
        return self.allocator.get_phy_allocator('cpu').get_num_free_blocks()

    def get_physical_blocks(self, logic_blocks: np.ndarray):
        """Get physical blocks."""
        return self.allocator.get_physical_blocks(logic_blocks)


def build_block_manager(cache_config: CacheConfig) -> BlockManager:
    """build block manager.

    Args:
        cache_config (CacheConfig):  cache_config.
    """

    num_cpu_blocks = cache_config.num_cpu_blocks
    num_gpu_blocks = cache_config.num_gpu_blocks

    return BlockManager(num_gpu_blocks, num_cpu_blocks)
