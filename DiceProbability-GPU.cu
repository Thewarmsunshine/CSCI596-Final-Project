#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.cuh"

using namespace std;

// 简单的伪随机数生成器函数
__device__ unsigned int simpleRand(unsigned int seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

// mapper 函数: 为每个输入元素生成一个骰子面 (1到6)
__device__ void mapper(input_type *input, Pair *pairs) {
    // We set the key of each input to 0.
    unsigned int seed = threadIdx.x + blockIdx.x * blockDim.x;
    seed = simpleRand(seed);  // 使用简单的伪随机数生成器
    pairs->value = (seed % 6) + 1;  // 生成 1 到 6 之间的随机数

}

// reducer 函数: 统计每个骰子面出现的次数，并计算概率
__device__ void reducer(Pair *pairs, int len, output_type *output) {
    int counts[6] = {0};  // 用于统计每个面出现的次数
    for (int i = 0; i < len; i++) {
        int face = static_cast<int>(pairs[i].value);
        if (face >= 1 && face <= 6) {
            counts[face - 1]++;
        }
    }
	for (int i = 0; i < 6; i++) {
        output[i] = static_cast<float>(counts[i]) / len;  // 计算每个面的概率
    }
}

// 主函数: 运行 MapReduce 作业并输出结果
int main(int argc, char const *argv[]) {
    clock_t start, end;
    double cpu_time_used;

    start = clock(); // 开始计时
    
    // 分配内存
    size_t input_size = NUM_INPUT * sizeof(input_type);
    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    input_type *input = (input_type *)malloc(input_size);
    output_type *output = (output_type *)malloc(output_size);

    // 运行 MapReduce 作业
    runMapReduce(input, output);

    // 输出所有面的概率
    for (int i = 0; i < NUM_OUTPUT; i++) {
        printf("Probability of Face %d: %f%\n", i + 1, output[i] * 100);
    }

    end = clock(); // 结束计时
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; // 计算运行时间

    printf("Time taken: %f seconds\n", cpu_time_used); // 打印运行时间

    // 释放内存
    free(input);
    free(output);

    return 0;
}
