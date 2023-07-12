#pragma once
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>

namespace fastertransformer {

template<typename ElementI_, typename ElementO_, typename CacheShape_, int Threads, class Enable = void>
class Dequantizer;

template<typename ElementI_, typename ElementO_, typename CacheShape_, int Threads>
class Dequantizer<
    ElementI_,
    ElementO_,
    CacheShape_,
    Threads,
    std::enable_if_t<!std::is_same<typename std::decay_t<ElementI_>, typename std::decay_t<ElementO_>>::value>> {
public:
    static int const kCpAsyncBits = 32;
    static int const kThreads     = Threads;
    static int const kAdvanceRank = 0;

    using ElementI = ElementI_;
    static_assert(cutlass::sizeof_bits<ElementI>::value <= 8);
    using ElementO = typename std::
        conditional_t<std::is_same<half, typename std::decay<ElementO_>::type>::value, cutlass::half_t, ElementO_>;
    using CacheShape = CacheShape_;

    using Layout    = cutlass::layout::RowMajor;
    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
        cutlass::layout::PitchLinearShape<CacheShape::kColumn, CacheShape::kRow>,
        Threads,
        kCpAsyncBits / cutlass::sizeof_bits<ElementI>::value>;
    using AccessTypeI =
        cutlass::AlignedArray<ElementI,
                              ThreadMap::kElementsPerAccess,
                              (ThreadMap::kElementsPerAccess * cutlass::sizeof_bits<ElementI>::value / 8)>;
    using AccessTypeO =
        cutlass::AlignedArray<ElementO,
                              ThreadMap::kElementsPerAccess,
                              (ThreadMap::kElementsPerAccess * cutlass::sizeof_bits<ElementO>::value / 8)>;

    using IteratorI = cutlass::transform::threadblock::
        PredicatedTileAccessIterator<CacheShape, ElementI, Layout, kAdvanceRank, ThreadMap, AccessTypeI, false>;
    using IteratorO = cutlass::transform::threadblock::
        PredicatedTileAccessIterator<CacheShape, ElementO, Layout, kAdvanceRank, ThreadMap, AccessTypeO, false>;

public:
    CUTLASS_HOST_DEVICE
    Dequantizer() {}

    static CUTLASS_DEVICE int16_t thread_id()
    {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }

    CUTLASS_DEVICE
    void operator()(IteratorI& iterator_i, IteratorO& iterator_o, float scale_)
    {
        iterator_i.set_iteration_index(0);
        iterator_o.set_iteration_index(0);

#pragma unroll
        for (int count = 0; count < ThreadMap::Iterations::kCount; ++count) {
            if (iterator_i.valid()) {
                auto input_data  = *iterator_i.get();
                auto output_data = *iterator_o.get();

                constexpr int kElements = AccessTypeI::kElements;
#pragma unroll
                for (int elem_id = 0; elem_id < kElements; ++elem_id) {
                    const int data       = int(input_data[elem_id]);
                    float     f_data     = float(data) * scale_;
                    output_data[elem_id] = ElementO(f_data);
                }

                *iterator_o.get() = output_data;
            }
            iterator_i++;
            iterator_o++;
        }
    }
};

template<typename ElementI_, typename ElementO_, typename CacheShape_, int Threads>
class Dequantizer<
    ElementI_,
    ElementO_,
    CacheShape_,
    Threads,
    std::enable_if_t<std::is_same<typename std::decay_t<ElementI_>, typename std::decay_t<ElementO_>>::value>> {
public:
    using ElementI = typename std::
        conditional_t<std::is_same<half, typename std::decay<ElementI_>::type>::value, cutlass::half_t, ElementI_>;
    ;
    using ElementO   = ElementI;
    using CacheShape = CacheShape_;

    using IteratorI = int;
    using IteratorO = int;

public:
    CUTLASS_HOST_DEVICE
    Dequantizer() {}

    static CUTLASS_DEVICE int16_t thread_id()
    {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }

    CUTLASS_DEVICE
    void operator()(IteratorI& iterator_i, IteratorO& iterator_o, float scale_)
    {
        assert(false);
    }
};

template<typename Dequantizer, bool kNeedDequant>
class dequantize;

template<typename Dequantizer>
class dequantize<Dequantizer, true> {
public:
    static CUTLASS_DEVICE int16_t thread_id()
    {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }

    CUTLASS_DEVICE dequantize(typename Dequantizer::ElementO** out_pptr,
                              typename Dequantizer::ElementI*  in_ptr,
                              int                              stride_i,
                              const cutlass::MatrixCoord&      extent_i,
                              const cutlass::MatrixCoord&      offset_i,
                              int                              stride_o,
                              const cutlass::MatrixCoord&      extent_o,
                              const cutlass::MatrixCoord&      offset_o,
                              float                            scale)
    {

        typename Dequantizer::IteratorI iterator_i(
            {Dequantizer::Layout(stride_i)}, in_ptr, extent_i, thread_id(), offset_i);
        typename Dequantizer::IteratorO iterator_o(
            {Dequantizer::Layout(stride_o)}, *out_pptr, extent_o, thread_id(), offset_o);

        Dequantizer dequantizer;
        dequantizer(iterator_i, iterator_o, scale);
    }
};

template<typename Dequantizer>
class dequantize<Dequantizer, false> {
public:
    CUTLASS_DEVICE dequantize(typename Dequantizer::ElementO** out_pptr,
                              typename Dequantizer::ElementI*  in_ptr,
                              int                              stride_i,
                              const cutlass::MatrixCoord&      extent_i,
                              const cutlass::MatrixCoord&      offset_i,
                              int                              stride_o,
                              const cutlass::MatrixCoord&      extent_o,
                              const cutlass::MatrixCoord&      offset_o,
                              float                            scale)
    {
        *out_pptr = in_ptr;
    }
};

}  // namespace fastertransformer