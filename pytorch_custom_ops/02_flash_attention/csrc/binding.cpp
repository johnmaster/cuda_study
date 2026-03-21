#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> flash_attn_forward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor> flash_attn_backward_cuda(
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L);

std::vector<torch::Tensor> flash_attn_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    return flash_attn_forward_cuda(Q, K, V);
}

std::vector<torch::Tensor> flash_attn_backward(
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L) {
    return flash_attn_backward_cuda(dO, Q, K, V, O, L);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &flash_attn_forward,  "Flash Attention forward  (CUDA)");
    m.def("backward", &flash_attn_backward, "Flash Attention backward (CUDA)");
}
