#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

///////////////////////////////////////////////////////////////
__global__ void qk_matmul_kernel(
    const __half* __restrict__ Q,   // 使用__half类型
    const __half* __restrict__ K,
    __half* output,
    int d_model,
    int seq_len_k
) {
    // 索引计算
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int row = threadIdx.x;

    extern __shared__ __half K_shared[]; // 共享内存类型为__half

    // 偏移计算
    const int q_offset = batch_idx * gridDim.y * seq_len_k * d_model 
                       + head_idx * seq_len_k * d_model;
    const int k_offset = batch_idx * gridDim.y * seq_len_k * d_model 
                       + head_idx * seq_len_k * d_model;

    // 分块计算
    const int tile_size = blockDim.x;
    const int num_tiles = (d_model + tile_size - 1) / tile_size;
    float acc = 0.0f;

    for (int t = 0; t < num_tiles; ++t) {
        // 加载K到共享内存（显式类型转换）
        const int k_col = t * tile_size + threadIdx.x;
        if (k_col < d_model) {
            for (int n = 0; n < seq_len_k; ++n) {
                K_shared[n * tile_size + threadIdx.x] = 
                    __float2half(__half2float(K[k_offset + n * d_model + k_col]));
            }
        }
        __syncthreads();

        // 计算Q*K^T
        const int q_col = t * tile_size + threadIdx.x;
        if (q_col < d_model) {
            const float q_val = __half2float(Q[q_offset + row * d_model + q_col]);
            for (int n = 0; n < seq_len_k; ++n) {
                const float k_val = __half2float(K_shared[n * tile_size + threadIdx.x]);
                acc += q_val * k_val;
            }
        }
        __syncthreads();
    }

    // 写入结果（转换为__half）
    if (row < seq_len_k) {
        const int out_idx = batch_idx * (gridDim.y * seq_len_k * seq_len_k)
                          + head_idx * (seq_len_k * seq_len_k)
                          + row * seq_len_k;
        output[out_idx] = __float2half_rn(acc);
    }
}

torch::Tensor qk_matmul_cuda(torch::Tensor Q, torch::Tensor K) {
    // 获取维度信息
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len_q = Q.size(2);
    const int seq_len_k = K.size(2);
    const int d_model = Q.size(3);

    auto options = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
    auto output = torch::zeros({batch_size, num_heads, seq_len_q, seq_len_k}, options);

    // 核函数配置
    dim3 blocks(batch_size, num_heads);
    dim3 threads(32);
    size_t shared_mem_size = seq_len_k * 32 * sizeof(__half);

    // 启动核函数（显式指针转换）
    qk_matmul_kernel<<<blocks, threads, shared_mem_size>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        d_model,
        seq_len_k
    );

    return output;
}
///////////////////////////////////////////////////////////////

__global__ void softmax_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const int64_t* input_shape,
    int ndim,
    int dim,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = outer_size * inner_size;
    if (group_idx >= total_groups) return;

    int outer_idx = group_idx / inner_size;
    int inner_idx = group_idx % inner_size;

    // 计算当前组的最大值
    float max_val = -INFINITY;
    for (int i = 0; i < dim_size; ++i) {
        int input_idx = outer_idx * dim_size * inner_size + i * inner_size + inner_idx;
        float val = __half2float(input[input_idx]);
        max_val = fmaxf(max_val, val);
    }

    // 计算指数和
    float exp_sum = 0.0f;
    for (int i = 0; i < dim_size; ++i) {
        int input_idx = outer_idx * dim_size * inner_size + i * inner_size + inner_idx;
        float val = __half2float(input[input_idx]);
        exp_sum += expf(val - max_val);
    }

    // 计算并写入输出
    for (int i = 0; i < dim_size; ++i) {
        int input_idx = outer_idx * dim_size * inner_size + i * inner_size + inner_idx;
        float val = __half2float(input[input_idx]);
        float exp_val = expf(val - max_val);
        float softmax_val = exp_val / exp_sum;
        output[input_idx] = __float2half(softmax_val);
    }
}

torch::Tensor softmax_cuda(torch::Tensor input, int dim) {
    auto output = torch::empty_like(input);

    auto input_shape = input.sizes();
    int64_t ndim = input.dim();
    int64_t dim_size = input_shape[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= input_shape[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= input_shape[i];
    }

    // 分配并拷贝输入形状到设备
    int64_t* d_input_shape;
    cudaMalloc(&d_input_shape, ndim * sizeof(int64_t));
    cudaMemcpy(d_input_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    // 计算执行配置
    int total_groups = outer_size * inner_size;
    int threads_per_block = 256;
    int blocks = (total_groups + threads_per_block - 1) / threads_per_block;

    softmax_kernel<<<blocks, threads_per_block>>>(
        reinterpret_cast<const __half*>(input.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<torch::Half>()),
        d_input_shape,
        ndim,
        dim,
        outer_size,
        inner_size,
        dim_size
    );

    cudaFree(d_input_shape);
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