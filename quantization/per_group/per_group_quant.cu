#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void per_group_quantize_kernel(const float* input, int8_t* output,
                                           float* scales, int group_size, int n) {
    int group_id = blockIdx.x;
    int tid = threadIdx.x;
    int start = group_id * group_size;

    extern __shared__ float sdata[];

    // 找组内 absmax
    float local_max = 0.0f;
    for (int i = tid; i < group_size && (start + i) < n; i += blockDim.x) {
        float val = fabsf(input[start + i]);
        if (val > local_max) local_max = val;
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid] < sdata[tid + s])
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }

    float scale = sdata[0] / 127.0f;
    if (tid == 0) scales[group_id] = scale;
    __syncthreads();

    if (scale == 0.0f) scale = 1.0f;

    for (int i = tid; i < group_size && (start + i) < n; i += blockDim.x) {
        float val = input[start + i] / scale;
        val = fminf(fmaxf(val, -127.0f), 127.0f);
        output[start + i] = (int8_t)rintf(val);
    }
}

__global__ void per_group_dequantize_kernel(const int8_t* input, float* output,
                                             const float* scales, int group_size, int n) {
    int group_id = blockIdx.x;
    int tid = threadIdx.x;
    int start = group_id * group_size;

    float scale = scales[group_id];

    for (int i = tid; i < group_size && (start + i) < n; i += blockDim.x) {
        output[start + i] = (float)input[start + i] * scale;
    }
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
    const int N = 4096;
    const int BLOCK = 256;

    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) {
        float section_scale = (i / 128 + 1) * 0.1f;
        h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * section_scale;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int group_sizes[] = {32, 64, 128, 256};

    printf("===== Per-Group Quantization (varying group_size) =====\n\n");

    for (int gi = 0; gi < 4; gi++) {
        int gs = group_sizes[gi];
        int num_groups = (N + gs - 1) / gs;

        int8_t *d_quant;
        float *d_scales;
        cudaMalloc(&d_quant, N * sizeof(int8_t));
        cudaMalloc(&d_scales, num_groups * sizeof(float));

        per_group_quantize_kernel<<<num_groups, BLOCK, BLOCK * sizeof(float)>>>(
            d_input, d_quant, d_scales, gs, N);
        per_group_dequantize_kernel<<<num_groups, BLOCK>>>(
            d_quant, d_output, d_scales, gs, N);

        cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

        float mse = compute_mse(h_input, h_output, N);
        float overhead = num_groups * sizeof(float);
        printf("group_size=%3d  groups=%3d  MSE=%.8f  scale_overhead=%d bytes\n",
               gs, num_groups, mse, (int)overhead);

        cudaFree(d_quant);
        cudaFree(d_scales);
    }

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
