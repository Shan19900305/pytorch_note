# Basic Concept
从0开始学习基本概念

## 基于一个例子查看make_tiled_copy
  - Threads themselve are arranged in a (ThreadShape_M, ThreadShape_N) arrangement which is replicated over the tile;
  - Layout-of-Threads to describe the number and arrangement of threads (e.g. row-major, col-major, etc),
  - Layout-of-Values that each thread will access.
  - code：
    ```c++
      template <class... Args,
                class ThrLayout,
                class ValLayout = Layout<_1>>
      CUTE_HOST_DEVICE auto
      make_tiled_copy(Copy_Atom<Args...> const& copy_atom,
                      ThrLayout          const& thr_layout = {},     // (m,n) -> thr_idx
                      ValLayout          const& val_layout = {})     // (m,n) -> val_idx
    ```
  - example1： test/unit/cute/ampere/ldsm.cu
    ```c++
      struct SM75_U32x1_LDSM_N
      {
        using SRegisters = uint128_t[1];
        using DRegisters = uint32_t[1];

        CUTE_HOST_DEVICE static void
        copy(uint128_t const& smem_src,
            uint32_t& dst)
        {
      #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
          uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
          asm volatile ("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
              : "=r"(dst)
              :  "r"(smem_int_ptr));
      #else
          CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
      #endif
        }
      };

      template <>
      struct Copy_Traits<SM75_U32x1_LDSM_N>
      {
        // Logical thread id to thread idx (warp)
        using ThrID = Layout<_32>;

        // Map from (src-thr,src-val) to bit
        using SrcLayout = Layout<Shape <Shape <  _8,_4>,_128>,
                                Stride<Stride<_128,_0>,  _1>>;
        // Map from (dst-thr,dst-val) to bit
        using DstLayout = Layout<Shape <_32,_32>,
                                Stride<_32, _1>>;

        // Reference map from (thr,val) to bit
        using RefLayout = DstLayout;
      };

      auto smem_layout = Layout<Shape <_32,Shape <_2, _4>>,
                                Stride< _2,Stride<_1,_64>>>{};
      auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U32x1_LDSM_N, uint16_t>{},
                                        Layout<Shape<_32,_1>>{},
                                        Layout<Shape< _1,_8>>{});
      print_latex(tiled_copy);
    ```
    对应结果
    ```plain txt
      thr_layout: (_32,_1):(_1,_0)
      val_layout: (_1,_8):(_0,_1)
      layout_mn : (_32,_8):(_1,_32)
      layout_tv : (_32,_8):(_1,_32)
      tiler     : (_32,_8)
      % LayoutS: ((_4,_8),(_2,_4)):((_64,_1),(_32,_256))
      % ThrIDS : _32:_1
      % LayoutD: (_32,_8):(_1,_32)
      % ThrIDD : _32:_1
    ```

  - example2： examples/cute/tutorial/tiled_copy.cu
    ```c++
      auto tensor_shape = make_shape(256, 512);
      Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())),
                                    make_layout(tensor_shape));
      Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())),
                                    make_layout(tensor_shape));
      auto block_shape = make_shape(Int<128>{}, Int<64>{});
      Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n')
      Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n')
      // --> result: ((_128, _64), 2, 8):((_1, 256), _128, 16384)

      dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'
      dim3 blockDim(size(thr_layout));

      // Launch Kernel里面的工作
      // 对应取一块（128, 64）的数据
      Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)
      Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)

      // Thread arrangement
      Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));
      // Vector dimensions
      Layout vec_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));
      using CopyOp = UniversalCopy<uint_byte_t<sizeof(float) * _4{}>>;
      using Atom = Copy_Atom<CopyOp, float>;
      auto tiled_copy = make_tiled_copy(Atom{},
                                        thr_layout,
                                        vec_layout);
      // layout_mn : ((_4,_32),_8):((_256,_1),_32)
      // layout_tv : (_256,_4):(_4,_1)
      // tiler     : (_128,_8)
      // % LayoutS: ((_4,_32),_8):((_256,_1),_32)
      // % ThrIDS : _256:_1
      // % LayoutD: ((_4,_32),_8):((_256,_1),_32)
      // % ThrIDD : _256:_1
      // Construct a Tensor corresponding to each thread's slice.
      auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

      Tensor thr_tile_S = thr_copy.partition_S(tile_S);             // (CopyOp, CopyM, CopyN)
      Tensor thr_tile_D = thr_copy.partition_D(tile_D);             // (CopyOp, CopyM, CopyN)

      // Construct a register-backed Tensor with the same shape as each thread's partition
      // Use make_fragment because the first mode is the instruction-local mode
      Tensor fragment = make_fragment_like(thr_tile_D);             // (CopyOp, CopyM, CopyN)

      // Copy from GMEM to RMEM and from RMEM to GMEM
      copy(tiled_copy, thr_tile_S, fragment);
      copy(tiled_copy, fragment, thr_tile_D);
    ```

  - 参数解释：
    - Copy_Atom： 本例中为<SM75_U32x1_LDSM_N, uint16_t>{}
    - smem_layout表示一个数据块，对应为列模式存储。有32个小数据块，每个小数据块有8个数据。
      ```plain txt
          0  64 128 256
          1  65 129 257
          2  66 130 258
          3  67 131 259
         ...
         ...
         62 126 190 254
         63 127 191 255
      ```
    - ThrLayout：表示为线程布局。CUTLASS通过定义线程布局来决定如何在GPU上分配和管理线程。常见的布局方式包括线性布局（PitchLinear）和块状布局（Tiled Layout）。在`make_tiled_copy`中，线程布局决定了每个线程处理的数据块的形状和大小。本例中为Layout<Shape<_32,_1>>{}。
    - ValLayout:表示为数据块布局。在CUTLASS中，数据块布局决定了每个数据块处理的数据块的形状和大小。本例中为Layout<Shape< _1,_8>>{}。
    - 

参考资料：https://askai.glarity.app/zh-CN/search/%E5%A6%82%E4%BD%95%E5%9C%A8CUTLASS%E4%B8%AD%E4%BD%BF%E7%94%A8-make-tiled-copy-%E8%BF%9B%E8%A1%8C%E7%BA%BF%E7%A8%8B%E5%B8%83%E5%B1%80
参考资料：https://yichengdwu.github.io/MoYe.jl/dev/manual/tiled_matmul/