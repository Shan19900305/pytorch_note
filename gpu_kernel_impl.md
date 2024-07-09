# gpu_kernel_impl
模板函数，入参为TensorIteratorBase对象和func_t对象, 其中算子的实现由func_t对象决定，而输入输出的数据由TensorIteratorBase对象决定。
```c++
template <typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) { xxx }
```

## 函数实现
从函数功能实现上来看，该函数通过传入的TensorIteratorBase对象和func对象，通过一套通用模板完成kernel的实现。
从动态数据类型角度来看主要分成两块，即需要动态数据类型转换和不需要动态数据类型转换两个场景。该部分主要通过递归的方式实现，判断func_t中输入输出数据类型和TensorIteratorBase中输入输出数据类型是否一致，如果一致则调用gpu_kernel_impl_nocast，反之进行特殊处理。
从数据连续性角度来看，也可以分成两块，即支持连续数据处理（向量化处理）和离散数据处理两种场景。该部分主要通过TensorIteratorBase中is_contiguous()函数判断是否支持连续数据处理。


### needs_dynamic_casting
```c++
// For input types check.
template<typename func_t, int nargs=function_traits<func_t>::arity>
struct needs_dynamic_casting {
  static bool check(TensorIteratorBase& iter) {
    using traits = function_traits<func_t>;
    using cpp_type = typename traits::template arg<nargs - 1>::type;
    using cpp_map = c10::CppTypeToScalarType<cpp_type>;

    if (iter.input_dtype(nargs-1) != cpp_map::value) {
      return true;
    }
    return needs_dynamic_casting<func_t, nargs - 1>::check(iter);
  }
};
```

### 无数据类型转换场景
基于TensorIteratorBase获取处理输入输出内存指针地址，元素个数和连续性状态。

#### 连续场景
基于launch_vectorized_kernel函数实现，默认按照每个thread处理4个数的向量大小逻辑编写。其中向量大小支持4个数，2个数和1个数的判断处理。

  - can_vectorize_up_to： 通过递归的方式获取指针是否满足片上指针地址整除sizeof(T) * num数的要求；依次判断4 × sizeof(T）、2 × sizeof（T）和sizeof（T)。
  https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#data-transfer-between-host-and-device
  When a warp executes an instruction that accesses global memory, it coalesces the memory accesses of the threads within the warp into one or more of these memory transactions depending on the size of the word accessed by each thread and the distribution of the memory addresses across the threads.
  Global memory resides in device memory and device memory is accessed via 32-, 64-, or 128-byte memory transactions. These memory transactions must be naturally aligned: Only the 32-, 64-, or 128-byte segments of device memory that are aligned to their size (i.e., whose first address is a multiple of their size) can be read or written by memory transactions.

  - vectorized_elementwise_kernel： 


#### 非连续场景



### 数据类型转换场景


