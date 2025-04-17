#include <torch/extension.h>
#include "vit_ops_kernel.h"

torch::Tensor qk_matmul(torch::Tensor Q, torch::Tensor K) {
    CHECK_INPUT(Q); CHECK_INPUT(K);
    return qk_matmul_cuda(Q, K);
}

torch::Tensor optimized_softmax(torch::Tensor input) {
    CHECK_INPUT(input);
    return softmax_cuda(input,-1);
}

torch::Tensor fused_layer_norm(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    CHECK_INPUT(input); CHECK_INPUT(gamma); CHECK_INPUT(beta);
    return layer_norm_cuda(input, gamma, beta, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qk_matmul", &qk_matmul, "Optimized QK^T matrix multiplication");
    m.def("optimized_softmax", &optimized_softmax, "Optimized softmax");
    m.def("fused_layer_norm", &fused_layer_norm, "Fused LayerNorm");
}