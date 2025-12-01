#include "kernels.cuh"

int main() {
    CHECK_CUDA(cudaDeviceReset());

    std::vector<int> lst1 = {1024, 2048, 4096, 8192};
    std::vector<int> lst2 = {1024, 2048, 4096, 8192};

    for (auto l1 : lst1) {
        for (auto l2 : lst2) {
            benchMark(l1, l2);
        }
    }

    printf("all of testing completed\n");
    return 0;
}