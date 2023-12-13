#ifndef MAP_REDUCE_CUH
#define MAP_REDUCE_CUH

// 配置 GPU 参数
#define GRID_SIZE 1024 // 定义网格大小
#define BLOCK_SIZE 1024 // 定义块大小

// 设置输入元素的数量，输出元素的数量和每个输入元素的键值对数量
#define NUM_INPUT 10000000 // 定义输入元素的数量
#define NUM_OUTPUT 6  // 定义输出元素的数量（这里设为6，因为骰子有6个面）


// 自定义输入类型的示例
struct Coordinate {
    float x;   // x坐标
    float y;   // y坐标
};

// 设置输入、输出、键和值的类型
typedef Coordinate input_type; // 输入类型
typedef float output_type; // 输出类型（用于存储概率等浮点数）
typedef float value_type;  // 值的类型


struct Pair {
   value_type value; // 仅包含值的结构，用于MapReduce操作中的值传递
   // 定义操作符 '<' 用于比较两个 Pair 对象
    __host__ __device__
    bool operator<(const Pair& other) const {
        return value < other.value;
    }
};

void runMapReduce(input_type *input, output_type *output); // 运行 MapReduce 作业的函数

#endif // MAP_REDUCE_CUH
