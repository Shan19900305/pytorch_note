# pytorch IPC
Inter-Process Communication (IPC) 主要指允许多个进程访问和改变同一块内存。
pytorch基于IPC的功能，实现了Tensor内存块的共享，从而满足多个进程使用同一块内存空间。典型接口为：torch.Tensor.share_memory_，对应的官方解释为：Moves the underlying storage to shared memory. This is a no-op if the underlying storage is already in shared memory and for CUDA tensors. Tensors in shared memory cannot be resized.

torch初始化时，调用THPModule_initExtension->libshm_init函数传递默认路径，初始化全局变量manager_executable_path。