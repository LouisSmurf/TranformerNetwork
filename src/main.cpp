#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>


//relu
float relu(float x){return std::max(0.0f, x);}
float reluDerivative(float x){return (x > 0) ? 1.0f : 0.0f;}

//leaky relu
float leakyRelu(float x){return (x > 0) ? x : 0.01f * x;}
float leakyReluDerivative(float x){return (x > 0) ? 1.0f : 0.01f;}

struct MLPLayer{
    int inputSize; int outputSize;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::function<float(float)> activationFunction; 
    std::function<float(float)> activationDerivative;

    MLPLayer(int input, int output, std::function<float(float)> activation, std::function<float(float)> derivative) : inputSize(input), outputSize(output), activationFunction(activation), activationDerivative(derivative) {
        weights.resize(outputSize, std::vector<float>(inputSize));
        biases.resize(outputSize);
        
        initialiseWeights();
    }

    void initialiseWeights() {
        std::random_device random;
        std::mt19937 generateNumber(random());
        std::normal_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] = dist(generateNumber);
            }
            biases[i] = dist(generateNumber);
        }
    }

    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> output(outputSize, 0.0f);
        for (int i = 0; i < outputSize; ++i) {
            float activation = biases[i];
            for (int j = 0; j < inputSize; ++j) {
                activation += weights[i][j] * input[j];
            }
            output[i] = activationFunction(activation);
        }
        return output;
    }
};

struct MLP{
    std::vector<MLPLayer> layers;
    bool useSoftmaxOutput = false;

    MLP(std::vector<int> layerSizes, std::function<float(float)> activation = leakyRelu, std::function<float(float)> derivative = leakyReluDerivative) {
        for (int i = 0; i < layerSizes.size() - 1; ++i) {
            layers.push_back(MLPLayer(layerSizes[i], layerSizes[i + 1], activation, derivative));
        }
    }

    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> output = input;
        for (MLPLayer& layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }
};

int main() {
    MLP net({3, 5, 3}, leakyRelu, leakyReluDerivative, true);
    std::vector<float> result = net.forward({1.0f, 2.0f, 3.0f});
    for (float value : result) {
        std::cout << value << ", ";
    }
    return 0;
}