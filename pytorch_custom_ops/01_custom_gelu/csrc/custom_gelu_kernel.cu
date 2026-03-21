#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr float SQRT_2_OVER_PI = 0.7978845608028654f; // sqrt(2/pi)
constexpr float COEFF = 0.044715f;

__global__ void custom_gelu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float cube = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * cube);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void custom_gelu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float cube = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * cube);
        float tanh_val = tanhf(inner);
        float sech2 = 1.0f - tanh_val * tanh_val;
        float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x * x);

        // d/dx [0.5 * x * (1 + tanh(inner))]
        //   = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d_inner
        float grad = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
        grad_input[idx] = grad_output[idx] * grad;
    }
}

torch::Tensor custom_gelu_forward_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    int n = input.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    custom_gelu_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

torch::Tensor custom_gelu_backward_cuda(torch::Tensor grad_output, torch::Tensor input) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

    auto grad_input = torch::empty_like(input);
    int n = input.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    custom_gelu_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        n
    );

    return grad_input;
}
