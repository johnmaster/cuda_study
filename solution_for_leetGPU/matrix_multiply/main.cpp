#include "kernels.cuh"

int main() {
    CHECK_CUDA(cudaDeviceReset());
    
    std::vector<std::tuple<int, int, int>> shapes = {
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {6144, 6144, 6144},
        {8192, 8192, 8192}, 
    };

    for (auto [M, N, K] : shapes) {
        benchMark(M, N, K);
    }
    
    printf("all of testing completed\n");
    return 0;
}