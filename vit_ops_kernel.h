// vit_ops_kernel.h
#ifndef VIT_OPS_KERNEL_H
#define VIT_OPS_KERNEL_H

#include <torch/extension.h>  // PyTorch C++前端头文件
#include <cuda.h>
#include <cuda_runtime.h>

// ================= 核函数声明 =================
// QK^T矩阵乘法
torch::Tensor qk_matmul_cuda(
    torch::Tensor Q,    // 输入Query张量
    torch::Tensor K     // 输入Key张量
);

// 优化Softmax
torch::Tensor softmax_cuda(
    torch::Tensor input, // 输入张量
    int dim             // Softmax维度
);

// 融合LayerNorm
torch::Tensor layer_norm_cuda(
    torch::Tensor input,  // 输入张量
    torch::Tensor gamma,  // 缩放参数
    torch::Tensor beta,   // 平移参数
    float eps            // 数值稳定项
);

// ================= 宏定义 =================
// 检查Tensor是否在GPU上且为连续内存
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#endif // VIT_OPS_KERNEL_H