#include "kernels.cuh"

extern "C" void convolutionRowCPU(float* host_dst, float* host_src, float* host_kernel,
    int imageW, int imageH, int kernelR) {
    
    for (int y = 0; y < imageH; y++) {
        for (int x = 0; x < imageW; x++) {
            float sum = 0;
            
            for (int k = -kernelR; k <= kernelR; k++) {
                int d = k + x;
                
                if (d >= 0 && d < imageW)
                    sum += host_src[y * imageW + x] * host_kernel[kernelR - k];
            }
            host_dst[y * imageW + x] = sum;
        }
    }
}
extern "C" void convolutionColumnCPU(float* host_dst, float* host_src, float* host_kernel,
       int imageW, int imageH, int kernelR) {
    
    for (int y = 0; y < imageH; y++) {
        for (int x = 0; x < imageW; x++) {
            
            float sum = 0;
            for (int k = -kernelR; k <= kernelR; k++) {
                int d = y + k;
                
                if (d >= 0 && d < imageH) {
                    sum += host_src[d * imageW + x] * host_kernel[kernelR - k];
                }
            }
            
            host_dst[y * imageW + x] = sum;
        }
    }
}
int main() {
    std::cout << std::left << std::setw(15)
              << "Start convolutionSeparable" << "\n";
    
    float* host_kernel, *host_input, *host_buffer, *host_outputCPU, *host_outputGPU;
    float* device_input, *device_output, *device_buffer;
    float time = 0;
    
    const int imageW = 3072;
    const int imageH = 3072;
    const int iterations = 16;
    
    std::cout << std::left
              << "Image Width x Height = " << imageW  << " x " << imageH << "\n";
    
    host_kernel = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    host_input = (float *)malloc(imageW * imageH * sizeof(float));
    host_buffer = (float *)malloc(imageW * imageH * sizeof(float));
    host_outputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    host_outputGPU = (float *)malloc(imageW * imageH * sizeof(float));
    srand(200);

    
    for (unsigned int i = 0; i < 17; i++) {
        host_kernel[i] = (float)(rand() % 16);
    }
    
    for (unsigned int i = 0; i < imageW * imageH; i++) {
        host_input[i] = (float)(rand() % 16);
    }
    
    setConvolutionKernel(host_kernel);
    
    CHECK_CUDA(cudaMalloc(&device_input, imageH * imageW * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&device_output, imageH * imageW * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_buffer, imageH * imageW * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(device_input, host_input, imageH * imageW * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < 5; i++) {
        convolutionRowsGPU(device_buffer, device_input, imageW, imageH);
        convolutionColumnsGPU(device_output, device_buffer, imageW, imageH);
    }

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        convolutionRowsGPU(device_buffer, device_input, imageW, imageH);
        convolutionColumnsGPU(device_output, device_buffer, imageW, imageH);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
    
    std::cout << std::left << std::setw(15) << "convolution GPU "
              << std::setprecision(4) << (time / iterations) << " us\n";

    CHECK_CUDA(cudaMemcpy(host_outputGPU, device_output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

    convolutionRowCPU(host_buffer, host_input, host_kernel, imageW, imageH, KERNEL_RADIUS);
    convolutionColumnCPU(host_outputCPU, host_buffer, host_kernel, imageW, imageH, KERNEL_RADIUS);
    
    double sum = 0, delta = 0;
    for (unsigned int i = 0; i < imageW * imageH; i++) {
        delta += (host_outputGPU[i] - host_outputCPU[i]) * (host_outputGPU[i] - host_outputCPU[i]);
        sum += host_outputCPU[i] * host_outputCPU[i];
    }
    double l2norm = sqrt(delta / sum);
    printf("L2 norm: %E \n\n", l2norm);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(host_kernel);
    free(host_input);
    free(host_buffer);
    free(host_outputCPU);
    free(host_outputGPU);
    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));
    CHECK_CUDA(cudaFree(device_buffer));

    return 0;
}