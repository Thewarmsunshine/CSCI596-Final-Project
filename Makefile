CC = /usr/bin/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /spack/apps/linux-centos7-x86_64/gcc-9.2.0/cuda-10.2.89-knz2qolpsp7nkkvcwkdfv6bkmu3adpe4
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_61,code=compute_61

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifeq ($(shell uname),Darwin)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH)
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH)
      CCFLAGS   := -m32
  else
      CUDA_LIB_PATH := $(CUDA_LIB_PATH)64
      LDFLAGS       := -L$(CUDA_LIB_PATH)
      CCFLAGS       := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

TARGETS = DiceProbability-GPU

all: $(TARGETS)

#DiceProbability-GPU: map_reduce.cu DiceProbability-GPU.o ta_utilities.o
#	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ $^

DiceProbability-GPU: DiceProbability-GPU.cu map_reduce.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) --device-c $^
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ DiceProbability-GPU.o map_reduce.o

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)

run: $(TARGETS)
	./DiceProbability-GPU
