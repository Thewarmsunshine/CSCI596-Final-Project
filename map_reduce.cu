#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "config.cuh"

using namespace std;


// 从外部引入的 mapper 和 reducer 函数
extern __device__ void mapper(input_type *input, Pair *pairs);
extern __device__ void reducer(Pair *pairs, int len, output_type *output);

// GPU 错误检查宏
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code,
    const char *file,
    int line,
    bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// 声明 CUDA map 和 reduce 函数
void cudaMap(input_type *input, Pair *pairs);
void cudaReduce(Pair *pairs, output_type *output);

// map 内核函数：每个线程处理一个输入元素
__global__ void mapKernel(input_type *input, Pair *pairs) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < NUM_INPUT;
        i += blockDim.x * gridDim.x) {
        mapper(&input[i], &pairs[i]);
    }
}

// reduce 内核函数：处理排序后的 Pair 对象并运行 reducer
__global__ void reduceKernel(Pair *pairs, output_type *output) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < NUM_OUTPUT;
        i += blockDim.x * gridDim.x) {
        int valueSize = 0;

        valueSize = NUM_INPUT; // 每个输出都基于所有输入计算

        reducer(pairs, valueSize, &output[i]);
    }
}

// 主函数：运行整个 MapReduce 作业
void runMapReduce(input_type *input, output_type *output) {
    input_type   *dev_input;
    output_type  *dev_output;
    Pair *dev_pairs;

    // 计算内存大小
    size_t input_size = NUM_INPUT * sizeof(input_type);
    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    size_t pairs_size = NUM_INPUT * sizeof(Pair);

    // 在 GPU 上初始化内存
    cudaMalloc(&dev_input, input_size);
    cudaMalloc(&dev_pairs, pairs_size);

    // 将输入数据复制到 GPU
    cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);


    // 运行 map 内核
    cudaMap(dev_input, dev_pairs);

    // 使用 Thrust 对 Pair 对象进行排序
    thrust::device_ptr<Pair> dev_ptr(dev_pairs);
    thrust::sort(dev_ptr, dev_ptr + NUM_INPUT);

    // 释放输入数据所用的 GPU 空间
    cudaFree(dev_input);
    // 为输出数据分配 GPU 空间
    cudaMalloc(&dev_output, output_size);

    // 运行 reduce 内核
    cudaReduce(dev_pairs, dev_output);

    // 将输出数据复制回主机
    cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);

    // 释放 GPU 上的内存
    cudaFree(dev_pairs);
    cudaFree(dev_output);
}

// 调用 map 内核并检查错误
void cudaMap(input_type *input, Pair *pairs) {
    mapKernel<<<GRID_SIZE, BLOCK_SIZE>>>(input, pairs);
    gpuErrChk( cudaPeekAtLastError() );
    gpuErrChk( cudaDeviceSynchronize() );
}

// 调用 reduce 内核并检查错误
void cudaReduce(Pair *pairs, output_type *output) {
    reduceKernel<<<GRID_SIZE, BLOCK_SIZE>>>(pairs, output);
    gpuErrChk( cudaPeekAtLastError() );
    gpuErrChk( cudaDeviceSynchronize() );
}
