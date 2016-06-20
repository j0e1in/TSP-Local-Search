COMMON = ../common/city.c
ALG = sa

all: cpu cuda

cpu: $(ALG).c
	gcc $(ALG).c $(COMMON) -o $(ALG)_cpu

cuda: $(ALG)_cuda.cu
	nvcc $(ALG)_cuda.cu $(COMMON) -o $(ALG)_gpu

clean:
	rm -rf $(ALG)_cpu $(ALG)_gpu result.txt