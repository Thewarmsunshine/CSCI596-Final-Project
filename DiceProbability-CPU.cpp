#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>

using namespace std;

struct Pair {
    int value;  // 骰子面
};

// 生成随机骰子面的函数（mapper）
void mapper(vector<Pair>& pairs) {
    static mt19937 rng(static_cast<unsigned int>(time(nullptr)));
    uniform_int_distribution<mt19937::result_type> dist(1, 6);
    for (auto& pair : pairs) {
        pair.value = dist(rng);
    }
}

// 统计概率的函数（reducer）
vector<float> reducer(const vector<Pair>& pairs, int numFaces) {
    vector<int> counts(numFaces, 0);
    for (const auto& pair : pairs) {
        counts[pair.value - 1]++;
    }

    vector<float> probabilities;
    for (int count : counts) {
        probabilities.push_back(static_cast<float>(count) / pairs.size());
    }
    return probabilities;
}

int main() {
    const int numInputs = 100000000;
    vector<Pair> pairs(numInputs);

    auto start = chrono::high_resolution_clock::now();

    mapper(pairs);
    vector<float> probabilities = reducer(pairs, 6);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    for (int i = 0; i < 6; i++) {
        cout << "Probability of Face " << i + 1 << ": " << probabilities[i] * 100 << "%\n";
    }

    cout << "Time taken: " << elapsed.count() << " seconds";

    return 0;
}
