COMMON = "../common/city.c"
ALG = sa

all: cpu cuda

cpu: $(ALG).c
	gcc -Wall $(ALG).c $(COMMON) -o $(ALG)_cpu

cuda:
	nvcc $(ALG)_cuda.cu $(COMMON) -o $(ALG)_gpu

clean:
	rm -rf $(ALG)_cpu $(ALG)_gpu result.txt