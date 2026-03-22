#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void find_absmax_kernel(const float* input, float* absmax, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? fabsf(input[idx]) : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid] < sdata[tid + s]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)absmax, __float_as_int(sdata[0]));
    }
}

__global__ void symmetric_quantize_kernel(const float* input, int8_t* output,
                                          float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = input[idx] / scale;
    val = fminf(fmaxf(val, -127.0f), 127.0f);
    output[idx] = (int8_t)rintf(val);
}

__global__ void symmetric_dequantize_kernel(const int8_t* input, float* output,
                                            float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = (float)input[idx] * scale;
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
    for (int i = 0; i < N; i++) {
        h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    float *d_input, *d_output, *d_absmax;
    int8_t *d_quant;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_absmax, sizeof(float));
    cudaMalloc(&d_quant, N * sizeof(int8_t));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_absmax, 0, sizeof(float));

    // Step 1: 找 absmax
    find_absmax_kernel<<<GRID, BLOCK, BLOCK * sizeof(float)>>>(d_input, d_absmax, N);

    float h_absmax;
    cudaMemcpy(&h_absmax, d_absmax, sizeof(float), cudaMemcpyDeviceToHost);
    float scale = h_absmax / 127.0f;

    // Step 2: 量化
    symmetric_quantize_kernel<<<GRID, BLOCK>>>(d_input, d_quant, scale, N);

    // Step 3: 反量化
    symmetric_dequantize_kernel<<<GRID, BLOCK>>>(d_quant, d_output, scale, N);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("===== Symmetric INT8 Quantization =====\n");
    printf("absmax = %.6f, scale = %.6f\n", h_absmax, scale);
    printf("MSE = %.8f\n", compute_mse(h_input, h_output, N));
    printf("Memory: FP32 = %d bytes, INT8 = %d bytes (%.1fx compression)\n",
           (int)(N * sizeof(float)), (int)(N * sizeof(int8_t)),
           (float)sizeof(float) / sizeof(int8_t));
    int8_t* h_quant = (int8_t*)malloc(N * sizeof(int8_t));
    cudaMemcpy(h_quant, d_quant, N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    printf("\nSamples:\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d] original=%.6f  quantized=%4d  recovered=%.6f  error=%.6f\n",
               i, h_input[i], (int)h_quant[i], h_output[i],
               fabsf(h_input[i] - h_output[i]));
    }

    free(h_input);
    free(h_output);
    free(h_quant);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_absmax);
    cudaFree(d_quant);
    return 0;
}
