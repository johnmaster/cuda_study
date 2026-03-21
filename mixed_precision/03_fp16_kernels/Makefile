NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80

all: fp16_gemm_cuda

fp16_gemm_cuda: fp16_gemm_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcudart

clean:
	rm -f fp16_gemm_cuda

.PHONY: all clean
