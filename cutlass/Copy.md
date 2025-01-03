# Basic Concept
从0开始学习基本概念

## 基于一个例子查看make_tiled_copy
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
  - example： test/unit/cute/ampere/ldsm.cu
    ```plain txt
      auto smem_layout = Layout<Shape <_32,Shape <_2, _4>>,
                                Stride< _2,Stride<_1,_64>>>{};
      auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U32x1_LDSM_N, uint16_t>{},
                                        Layout<Shape<_32,_1>>{},
                                        Layout<Shape< _1,_8>>{});
      print_latex(tiled_copy);
    ```
  - 输出结果：
    ```plain txt
      
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