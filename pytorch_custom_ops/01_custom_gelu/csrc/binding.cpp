#include <torch/extension.h>

// Declarations of CUDA functions (defined in custom_gelu_kernel.cu)
torch::Tensor custom_gelu_forward_cuda(torch::Tensor input);
torch::Tensor custom_gelu_backward_cuda(torch::Tensor grad_output, torch::Tensor input);

// Dispatch: check input is CUDA, then call the CUDA implementation
torch::Tensor custom_gelu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    return custom_gelu_forward_cuda(input);
}

torch::Tensor custom_gelu_backward(torch::Tensor grad_output, torch::Tensor input) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    return custom_gelu_backward_cuda(grad_output, input);
}

// pybind11 module definition — this is where Python sees the functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &custom_gelu_forward,  "Custom GELU forward  (CUDA)");
    m.def("backward", &custom_gelu_backward, "Custom GELU backward (CUDA)");
}
