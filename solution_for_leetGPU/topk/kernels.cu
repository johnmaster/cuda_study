#include "kernels.cuh"

void rand_init(float* h_data, size_t n) {
    std::mt19937 gen(200);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (size_t i = 0; i < n; i++) {
        h_data[i] = dist(gen);
    }
}
float* alloc_host(size_t n) {
    return new float[n];
}
float* alloc_device(size_t n) {
    float* p;
    CHECK_CUDA(cudaMalloc(&p, n * sizeof(float)));
    return p;
}
void free_host(float* p) {
    delete[] p;
}
void free_device(float* p) {
    CHECK_CUDA(cudaFree(p));
}

// down direction
__device__ __forceinline__ void compare_and_swap_desc(float& a, float& b) {
    if (a < b) {
        float tmp = a;
        a = b;
        b = tmp;
    }
}
// up direction
__device__ __forceinline__ void compare_and_swap_asc(float& a, float& b) {
    if (a > b) {
        float tmp = a;
        a = b;
        b = tmp;
    }
}

__global__ void bitonic_merge_global(float* data, size_t n, int stage, int substage) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // every thread responsible for a pair of elements
    // calculate the distance between the two elements
    int distance = 1 << substage;
    
    // calculate the position of compare elements
    size_t pair_id = idx;
    size_t group_len = distance * 2;
    
    /*
    (pair_id / distance): calculate the group index
    (pair_id / distance) * group_len: calculate the start index of the group
    (pair_id / distance) * group_len + (pair_id % distance): calculate the left index
    */
    size_t left = (pair_id / distance) * group_len + (pair_id % distance);
    size_t right = left + distance;

    if (right >= n) return;

    size_t big_group = 1 << (stage + 1);

    // true represents descending direction
    bool dir = ((left / big_group) % 2 == 0);

    float lv = data[left];
    float rv = data[right];
    
    bool should_swap = dir ? (lv < rv) : (lv > rv);
    if (should_swap) {
        data[left] = rv;
        data[right] = lv;
    }
}

__host__ size_t next_power_of_2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}
void solve_bitonic_sort(float* input, float* output, size_t n, size_t k) {
    size_t padded_n = next_power_of_2(n);

    float* d_work;
    CHECK_CUDA(cudaMalloc(&d_work, padded_n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_work, input, n * sizeof(float), cudaMemcpyDeviceToDevice));

    if (padded_n > n) {
        std::vector<float> padding(padded_n - n, -INFINITY);
        CHECK_CUDA(cudaMemcpy(d_work + n, padding.data(),
                    (padded_n - n) * sizeof(float), cudaMemcpyHostToDevice));
    }

    int num_stages = 0;
    for (size_t tmp = padded_n; tmp > 1; tmp >>= 1) num_stages++;

    int threads_per_block = BLOCK_SIZE;
    size_t num_pairs = padded_n / 2;
    int num_blocks = (num_pairs + threads_per_block - 1) / threads_per_block;

    for (int stage = 0; stage < num_stages; stage++) {
        for(int substage = stage; substage >= 0; substage--) {
            bitonic_merge_global<<<num_blocks, threads_per_block>>>(
                d_work, padded_n, stage, substage
            );
            CHECK_CUDA(cudaGetLastError());
        }
    }

    CHECK_CUDA(cudaMemcpy(output, d_work, k * sizeof(float), cudaMemcpyDeviceToDevice));
    
    CHECK_CUDA(cudaFree(d_work));
}