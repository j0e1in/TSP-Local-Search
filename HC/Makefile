ALG = hc
COMMON = ../common/city.cpp
CUDA_COMMON = ../common/city_cuda.cpp
FLAGS = -std=c++11

all: cpu gpu

cpu: $(ALG).cpp
	g++ $(ALG).cpp $(FLAGS) $(COMMON) -o $(ALG)_cpu -lm

gpu: $(ALG).cu
	nvcc $(ALG).cu $(FLAGS) $(CUDA_COMMON) -o $(ALG)_gpu

clean:
	rm -rf $(ALG)_cpu $(ALG)_gpu result*