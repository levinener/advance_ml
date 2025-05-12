#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

///////////////////////////////////////////////////////////////
__global__ void qk_matmul_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* output,
    int d_model,
    int seq_len_q,
    int seq_len_k,
    int batch_size,
    int num_heads  // 添加num_heads参数
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z / num_heads;  // 正确拆分batch/head
    const int head_idx = blockIdx.z % num_heads;

    if (row >= seq_len_q || n >= seq_len_k) return;

    // 修正后的offset计算
    const int q_offset = batch_idx * (num_heads * seq_len_q * d_model) +
                       head_idx * (seq_len_q * d_model) +
                       row * d_model;

    const int k_offset = batch_idx * (num_heads * seq_len_k * d_model) +
                       head_idx * (seq_len_k * d_model) +
                       n * d_model;

    float acc = 0.0f;
    #pragma unroll 4  // 循环展开优化
    for (int d = 0; d < d_model; ++d) {
        acc += Q[q_offset + d] * K[k_offset + d];
    }

    // 修正后的输出索引
    const int out_idx = batch_idx * (num_heads * seq_len_q * seq_len_k) +
                       head_idx * (seq_len_q * seq_len_k) +
                       row * seq_len_k + n;

    if (out_idx < batch_size * num_heads * seq_len_q * seq_len_k) {  // 安全保护
        output[out_idx] = acc;
    }
}

torch::Tensor qk_matmul_cuda(torch::Tensor Q, torch::Tensor K) {
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len_q = Q.size(2);
    const int seq_len_k = K.size(2);
    const int d_model = Q.size(3);

    auto options = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
    auto output = torch::zeros({batch_size, num_heads, seq_len_q, seq_len_k}, options);

    dim3 block(16, 16);
    dim3 grid(
        (seq_len_q + block.x - 1) / block.x,
        (seq_len_k + block.y - 1) / block.y,
        batch_size * num_heads  // z维度保持batch*heads组合
    );

    qk_matmul_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        output.data_ptr<float>(),
        d_model,
        seq_len_q,
        seq_len_k,
        batch_size,
        num_heads  // 传递关键参数
    );

    return output;
}
///////////////////////////////////////////////////////////////

__global__ void optimized_softmax_kernel1(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size,
    int64_t dim_stride
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= outer_size * inner_size) return;

    int outer_idx = group_idx / inner_size;
    int inner_idx = group_idx % inner_size;

    // 使用实际 stride 计算起始位置
    const float* group_start = input + outer_idx * dim_size * dim_stride + inner_idx;
    float* output_start = output + outer_idx * dim_size * dim_stride + inner_idx;

    // 计算最大值
    float max_val = -INFINITY;
    for (int i = 0; i < dim_size; ++i) {
        float val = group_start[i * dim_stride];
        max_val = fmaxf(max_val, val);
    }

    // 计算指数和
    float sum_exp = 0.0f;
    for (int i = 0; i < dim_size; ++i) {
        float val = group_start[i * dim_stride] - max_val;
        sum_exp += expf(val);
    }

    // 写入结果
    for (int i = 0; i < dim_size; ++i) {
        float val = group_start[i * dim_stride] - max_val;
        output_start[i * dim_stride] = expf(val) / sum_exp;
    }
}
__global__ void optimized_softmax_kernel2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size,
    int64_t dim_stride
) {
    const int warp_size = 32;
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;
    int groups_per_block = blockDim.x / warp_size;
    int group_idx = blockIdx.x * groups_per_block + warp_id;

    if (group_idx >= outer_size * inner_size) return;

    int outer_idx = group_idx / inner_size;
    int inner_idx = group_idx % inner_size;

    const float* group_start = input + outer_idx * (inner_size * dim_size) + inner_idx * dim_size;
    float* output_start = output + outer_idx * (inner_size * dim_size) + inner_idx * dim_size;
    int elements_per_thread = (dim_size + warp_size - 1) / warp_size;

    // 计算每个线程的最大值
    float  max_val = -INFINITY;
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = lane_id + i * warp_size;
        if (idx < dim_size) {
            float val = group_start[idx];
            max_val = fmaxf(max_val, val);
        }
    }

    // Warp归约求全局最大值
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        float tmp = __shfl_down_sync(0xffffffff, max_val, offset);
        max_val = fmaxf(max_val, tmp);
    }
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // 计算每个线程的指数和
    float sum_exp = 0.0f;
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = lane_id + i * warp_size;
        if (idx < dim_size) {
            float val = group_start[idx ];
            sum_exp += expf(val - max_val);
        }
    }

    // Warp归约求全局指数和
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

    // 计算并写入归一化结果
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = lane_id + i * warp_size;
        if (idx < dim_size) {
            float val = group_start[idx ];
            float exp_val = expf(val - max_val);
            output_start[idx] = exp_val / sum_exp;
        }
    }
}

__global__ void optimized_softmax_kernel3(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size,
    int64_t dim_stride
) {
    extern __shared__ float shared[];
    const int warp_size = 32;
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;
    int groups_per_block = blockDim.x / warp_size;
    int group_idx = blockIdx.x * groups_per_block + warp_id;

    if (group_idx >= outer_size * inner_size) return;

    int outer_idx = group_idx / inner_size;
    int inner_idx = group_idx % inner_size;


    const float* group_start = input + outer_idx * (inner_size * dim_size) + inner_idx * dim_size;
    float* output_start = output + outer_idx * (inner_size * dim_size) + inner_idx * dim_size;

    int elements_per_thread = (dim_size + warp_size - 1) / warp_size;

    // 计算每个线程的最大值
    float max_val  = -INFINITY,sum_exp = 0.0f, bigger;
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = lane_id + i * warp_size;
        if (idx < dim_size) {
            bigger = fmaxf(max_val, group_start[idx]);
            sum_exp = sum_exp * expf(max_val - bigger) + expf(group_start[idx]-bigger);
            max_val=bigger;
        }
    }

    float offsetMax,offsetSum;
    

    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        //__syncwarp();
        offsetMax=__shfl_xor_sync(0xffffffff, max_val, offset);
        offsetSum = __shfl_xor_sync(0xffffffff, sum_exp, offset);
        if (offsetMax > max_val) {
                sum_exp *= expf(max_val - offsetMax);
                max_val = offsetMax;
            } else {
                offsetSum *= expf(offsetMax - max_val);
            }
            sum_exp += offsetSum;
    }

    // 计算并写入归一化结果
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = lane_id + i * warp_size;
        if (idx < dim_size) {
            output_start[idx ] = expf(group_start[idx] - max_val) / sum_exp;
        }
    }
}

__global__ void optimized_softmax_kernel4(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size,
    int64_t dim_stride
){
    extern __shared__ float shared[];
    float* s_data = shared;  // 共享内存存储输入数据
    float* s_max = (float*)&s_data[dim_size]; // 存储归约的中间最大值

    const int tid = threadIdx.x;
    const int group_idx = blockIdx.x;

    if (group_idx >= outer_size * inner_size) return;

    int outer_idx = group_idx / inner_size;
    int inner_idx = group_idx % inner_size;

    const float* group_start = input + outer_idx * (inner_size * dim_size) + inner_idx * dim_size;
    float* output_start = output + outer_idx * (inner_size * dim_size) + inner_idx * dim_size;

    // 每个线程处理多个元素，加载到共享内存
    int elements_per_thread = (dim_size + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < dim_size) {
            s_data[idx] = group_start[idx];
        }
    }
    __syncthreads(); // 确保所有数据加载完成

    // Step 1: 计算每个线程的局部最大值
    float max_val = -INFINITY;
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < dim_size) {
            max_val = fmaxf(max_val, s_data[idx]);
        }
    }

    // Step 2: 块内归约求全局最大值
    // Warp级归约
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    // Block级归约（每个warp的最大值存入共享内存）
    if (tid % warpSize == 0) {
        s_max[tid / warpSize] = max_val;
    }
    __syncthreads();
    // 第一个warp负责归约所有warp的最大值
    if (tid < warpSize) {
        max_val = (tid < blockDim.x / warpSize) ? s_max[tid] : -INFINITY;
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
        }
    }
    __syncthreads();
    float global_max = __shfl_sync(0xFFFFFFFF, max_val, 0);

    // Step 3: 计算指数和
    float sum_exp = 0.0f;
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < dim_size) {
            sum_exp += expf(s_data[idx] - global_max);
        }
    }

    // Step 4: 块内归约求全局指数和
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }
    if (tid % warpSize == 0) {
        s_max[tid / warpSize] = sum_exp;
    }
    __syncthreads();
    if (tid < warpSize) {
        sum_exp = (tid < blockDim.x / warpSize) ? s_max[tid] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset >>= 1) {
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
        }
    }
    __syncthreads();
    float global_sum = __shfl_sync(0xFFFFFFFF, sum_exp, 0);

    // Step 5: 计算并写入结果
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < dim_size) {
            output_start[idx] = expf(s_data[idx] - global_max) / global_sum;
        }
    }
}


torch::Tensor softmax_cuda(torch::Tensor input, int dim) {
    auto output = torch::empty_like(input);
    
    const int64_t dim_size = input.size(dim);
    const int64_t dim_stride = input.stride(dim);
    
    // 计算实际 outer/inner size
    const int64_t outer_size = input.numel() / (dim_size * dim_stride);
    const int64_t inner_size = dim_stride;

    const int total_groups = outer_size * inner_size;
    const int threads = 256;
    const int blocks = (total_groups + threads - 1) / threads;

    optimized_softmax_kernel1<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        outer_size,
        inner_size,
        dim_size,
        dim_stride
    );
    
    cudaDeviceSynchronize();
    return output;
}
//////////////////////////////////////////////////////////////
__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float eps,
                                  int batch_size, int hidden_size, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * hidden_size) {
        int batch = idx / hidden_size;
        int h = idx % hidden_size;

        // Calculate mean and variance for this layer
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float val = input[batch * hidden_size + i];
            sum += val;
            sq_sum += val * val;
        }
        float mean = sum / hidden_size;
        float var = sq_sum / hidden_size - mean * mean;

        // Normalize and scale
        float norm = (input[idx] - mean) / sqrtf(var + eps);
        output[idx] = gamma[h] * norm + beta[h];
    }
}

torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);
    auto output = torch::zeros_like(input);

    const int threads_per_block = 256;
    const int blocks_per_grid = (batch_size * hidden_size + threads_per_block - 1) / threads_per_block;

    layer_norm_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        eps,
        batch_size,
        hidden_size,
        output.data_ptr<float>()
    );

    return output;
}