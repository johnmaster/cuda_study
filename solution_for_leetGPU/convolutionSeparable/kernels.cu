#include "kernels.cuh"

__constant__ float c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(float* host_kernel) {
    CHECK_CUDA( cudaMemcpyToSymbol(c_Kernel, host_kernel, KERNEL_LENGTH * sizeof(float)));
}

__global__ void convolutionRowsKernel(float* device_dst, float* device_src,
                                      int imageW, int imageH, int pitch) {
    __shared__ float s_data[ROW_BLOCKDIM_Y][(ROW_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROW_BLOCKDIM_X];
    
    const int baseX = (blockIdx.x * ROW_RESULT_STEPS - ROWS_HALO_STEPS) * ROW_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROW_BLOCKDIM_Y + threadIdx.y;

    device_src += baseY * pitch + baseX;
    device_dst += baseY * pitch + baseX;
    
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROW_RESULT_STEPS; i++) {
        s_data[threadIdx.y][threadIdx.x + i * ROW_BLOCKDIM_X] = device_src[i * ROW_BLOCKDIM_X];
    }

    for (int i = 0; i < ROWS_HALO_STEPS; i++) {
        s_data[threadIdx.y][threadIdx.x + i * ROW_BLOCKDIM_X] =
            (baseX >= -i * ROW_BLOCKDIM_X) ? device_src[i * ROW_BLOCKDIM_X] : 0;
    }

    for (int i = ROWS_HALO_STEPS + ROW_RESULT_STEPS; i < ROWS_HALO_STEPS + ROW_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
        s_data[threadIdx.y][threadIdx.x + i * ROW_BLOCKDIM_X] = 
            (imageW - baseX > i * ROW_BLOCKDIM_X) ? device_src[i * ROW_BLOCKDIM_X] : 0;
    }
    __syncthreads();

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROW_RESULT_STEPS; i++) {
        float sum = 0;
        
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_data[threadIdx.y][threadIdx.x + i * ROW_BLOCKDIM_X - j];
        }
        device_dst[i * ROW_BLOCKDIM_X] = sum;
    }
}
extern "C" void convolutionRowsGPU(float* device_dst, float* device_src,
                                   int imageW, int imageH) {
    dim3 blocks(imageW / (ROW_BLOCKDIM_X * ROW_RESULT_STEPS), imageH / ROW_BLOCKDIM_Y);
    dim3 threads(ROW_BLOCKDIM_X, ROW_BLOCKDIM_Y);

    convolutionRowsKernel<<<blocks, threads>>>(device_dst, device_src, imageW, imageH, imageW);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void convolutionColumnsKernel(float* device_dst, float* device_src,
                                         int imageW, int imageH, int pitch) {
    __shared__ float s_data[COLUMNS_BLOCKDIM_X]
                           [(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y];
    
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    
    device_src += baseY * pitch + baseX;
    device_dst += baseY * pitch + baseX;

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        s_data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = device_src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
        s_data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? device_src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS;
         i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
        s_data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 
            (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? device_src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }
    __syncthreads();

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        float sum = 0;

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }
        device_dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}

extern "C" void convolutionColumnsGPU(float* device_dst, float* device_src,
                                      int imageW, int imageH) {
    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel<<<blocks, threads>>>(device_dst, device_src, imageW, imageH, imageW);
    CHECK_CUDA(cudaGetLastError());
}
