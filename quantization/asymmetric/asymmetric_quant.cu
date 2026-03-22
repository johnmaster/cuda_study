#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

__global__ void asymmetric_quantize_kernel(const float* input, uint8_t* output,
                                            float scale, int zero_point, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = input[idx] / scale + (float)zero_point;
    val = fminf(fmaxf(val, 0.0f), 255.0f);
    output[idx] = (uint8_t)rintf(val);
}

__global__ void asymmetric_dequantize_kernel(const uint8_t* input, float* output,
                                              float scale, int zero_point, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = ((float)input[idx] - (float)zero_point) * scale;
}

float compute_mse(const float* a, const float* b, int n) {
    float mse = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        mse += diff * diff;
    }
    return mse / n;
}

int main() {
    const int N = 1024;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;

    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));
    srand(42);

    // 模拟 ReLU 后的激活值：全为正数，非对称分布
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 6.0f;  // [0, 6]
    }

    // 在 CPU 上计算 scale 和 zero_point
    float x_min = h_input[0], x_max = h_input[0];
    for (int i = 1; i < N; i++) {
        if (h_input[i] < x_min) x_min = h_input[i];
        if (h_input[i] > x_max) x_max = h_input[i];
    }
    float scale = (x_max - x_min) / 255.0f;
    int zero_point = (int)roundf(-x_min / scale);

    float *d_input, *d_output;
    uint8_t *d_quant;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_quant, N * sizeof(uint8_t));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    asymmetric_quantize_kernel<<<GRID, BLOCK>>>(d_input, d_quant, scale, zero_point, N);
    asymmetric_dequantize_kernel<<<GRID, BLOCK>>>(d_quant, d_output, scale, zero_point, N);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    uint8_t* h_quant = (uint8_t*)malloc(N * sizeof(uint8_t));
    cudaMemcpy(h_quant, d_quant, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    printf("===== Asymmetric UINT8 Quantization =====\n");
    printf("Data range: [%.4f, %.4f]\n", x_min, x_max);
    printf("scale = %.6f, zero_point = %d\n", scale, zero_point);
    printf("MSE = %.8f\n", compute_mse(h_input, h_output, N));
    printf("\nSamples:\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d] original=%.6f  quantized=%3u  recovered=%.6f  error=%.6f\n",
               i, h_input[i], (unsigned)h_quant[i], h_output[i],
               fabsf(h_input[i] - h_output[i]));
    }

    free(h_input); free(h_output); free(h_quant);
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_quant);
    return 0;
}
