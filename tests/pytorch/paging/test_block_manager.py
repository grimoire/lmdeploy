import pytest

from lmdeploy.pytorch.paging.block_manager import (BlockManager,
                                                   LogicalAllocator)


class TestAllocator:

    @pytest.fixture
    def num_gpu_blocks(self):
        yield 16

    @pytest.fixture
    def num_cpu_blocks(self):
        yield 4

    @pytest.fixture
    def allocator(self, num_cpu_blocks, num_gpu_blocks):
        yield LogicalAllocator(num_cpu_blocks, num_gpu_blocks)

    def test_alloc(self, allocator, num_cpu_blocks, num_gpu_blocks):

        # initialize
        num_blocks = num_cpu_blocks + num_gpu_blocks
        gpu_allocator = allocator.get_phy_allocator('gpu')
        cpu_allocator = allocator.get_phy_allocator('cpu')
        assert allocator.get_num_free_blocks() == num_blocks
        assert cpu_allocator.get_num_free_blocks() == num_cpu_blocks
        assert gpu_allocator.get_num_free_blocks() == num_gpu_blocks

        # test allocate
        block_size = 4
        blocks = allocator.allocate(block_size, 'gpu')
        assert len(blocks) == block_size
        assert allocator.get_num_free_blocks() == num_blocks - block_size
        assert gpu_allocator.get_num_free_blocks(
        ) == num_gpu_blocks - block_size

        # test free
        allocator.free(blocks[:2])
        assert allocator.get_num_free_blocks() == num_blocks - 2
        assert gpu_allocator.get_num_free_blocks() == num_gpu_blocks - 2

        with pytest.raises(AssertionError):
            allocator.free(blocks)

    def test_full(self, allocator, num_cpu_blocks, num_gpu_blocks):

        num_blocks = num_cpu_blocks + num_gpu_blocks
        gpu_allocator = allocator.get_phy_allocator('gpu')
        cpu_allocator = allocator.get_phy_allocator('cpu')

        # no free blocks
        gpu_block_size = num_gpu_blocks
        gpu_blocks = allocator.allocate(gpu_block_size, 'gpu')
        cpu_block_size = num_cpu_blocks
        cpu_blocks = allocator.allocate(cpu_block_size, 'cpu')
        assert cpu_allocator.get_num_free_blocks() == 0
        assert gpu_allocator.get_num_free_blocks() == 0
        with pytest.raises(AssertionError):
            allocator.allocate(1, 'gpu')
        allocator.free(gpu_blocks)
        allocator.free(cpu_blocks)
        assert allocator.get_num_free_blocks() == num_blocks
        assert gpu_allocator.get_num_free_blocks() == num_gpu_blocks
        assert cpu_allocator.get_num_free_blocks() == num_cpu_blocks


class TestBlockManager:

    @pytest.fixture
    def num_cpu_blocks(self):
        yield 4

    @pytest.fixture
    def num_gpu_blocks(self):
        yield 4

    @pytest.fixture
    def block_mgr(self, num_cpu_blocks, num_gpu_blocks):
        yield BlockManager(num_cpu_blocks, num_gpu_blocks)

    def test_alloc(self, block_mgr, num_gpu_blocks):

        # test alloc
        num_blocks = 1
        assert block_mgr.can_allocate(num_blocks)
        blocks = block_mgr.allocate(num_blocks)
        phy_blocks = block_mgr.get_physical_blocks(blocks)
        assert block_mgr.get_num_free_gpu_blocks() == (num_gpu_blocks -
                                                       num_blocks)
        assert phy_blocks is not None
        assert len(phy_blocks) == num_blocks

        # test free
        block_mgr.free(blocks, 'gpu')
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks

        # alloc over limit
        num_blocks = num_gpu_blocks + 1
        assert not block_mgr.can_allocate(num_blocks)

    def test_swap(self, block_mgr, num_cpu_blocks, num_gpu_blocks):
        num_blocks = 2
        blocks = block_mgr.allocate(num_blocks)

        old_phy_blocks = block_mgr.get_physical_blocks(blocks)
        success = block_mgr.can_swap_out(blocks)
        swap_map = block_mgr.swap_out(blocks)
        new_phy_blocks = block_mgr.get_physical_blocks(blocks)
        assert success
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks
        assert block_mgr.get_num_free_cpu_blocks() == (num_gpu_blocks -
                                                       num_blocks)
        assert len(swap_map) == num_blocks
        for block_id in old_phy_blocks:
            assert block_id in swap_map
        for block_id in new_phy_blocks:
            assert block_id - num_gpu_blocks in swap_map.values()

        old_phy_blocks = block_mgr.get_physical_blocks(blocks)
        success = block_mgr.can_swap_in(blocks)
        swap_map = block_mgr.swap_in(blocks)
        new_phy_blocks = block_mgr.get_physical_blocks(blocks)
        assert block_mgr.get_num_free_gpu_blocks() == (num_gpu_blocks -
                                                       num_blocks)
        assert block_mgr.get_num_free_cpu_blocks() == num_gpu_blocks
        assert len(swap_map) == 2
        for block_id in old_phy_blocks:
            assert block_id - num_gpu_blocks in swap_map
        for block_id in new_phy_blocks:
            assert block_id in swap_map.values()

        success = block_mgr.can_swap_out(blocks)
        assert success
        num_blocks = num_cpu_blocks
        blocks = block_mgr.allocate(num_blocks, 'cpu')
        success = block_mgr.can_swap_in(blocks)
        assert not success
