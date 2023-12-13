#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>

struct Pair {
    int value;  // 骰子面
};

// 生成随机骰子面的函数（mapper）
void mapper(std::vector<Pair>& pairs) {
    static std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, 6);
    for (auto& pair : pairs) {
        pair.value = dist(rng);
    }
}

// 统计概率的函数（reducer）
std::vector<float> reducer(const std::vector<Pair>& pairs, int numFaces) {
    std::vector<int> counts(numFaces, 0);
    for (const auto& pair : pairs) {
        counts[pair.value - 1]++;
    }

    std::vector<float> probabilities;
    for (int count : counts) {
        probabilities.push_back(static_cast<float>(count) / pairs.size());
    }
    return probabilities;
}

int main() {
    const int numInputs = 100000000;
    std::vector<Pair> pairs(numInputs);

    auto start = std::chrono::high_resolution_clock::now();

    mapper(pairs);
    std::vector<float> probabilities = reducer(pairs, 6);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    for (int i = 0; i < 6; i++) {
        std::cout << "Probability of Face " << i + 1 << ": " << probabilities[i] * 100 << "%\n";
    }

    std::cout << "Time taken: " << elapsed.count() << " seconds";

    return 0;
}
