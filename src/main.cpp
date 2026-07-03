#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>

struct ActivationFunction{
    std::function<float(float)> compute;
    std::function<float(float)> computeDerivative;
    ActivationFunction(std::function<float(float)> f, std::function<float(float)> df) : compute(f), computeDerivative(df) {}
};

struct LossFunction{
    std::function<float(const std::vector<float>&, const std::vector<float>&)> compute;
    std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)> computeDerivative;
    LossFunction(std::function<float(const std::vector<float>&, const std::vector<float>&)> f, std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)> df) : compute(f), computeDerivative(df) {}
};

//relu
float relu(float x){return std::max(0.0f, x);}
float reluDerivative(float x){return (x > 0) ? 1.0f : 0.0f;}
ActivationFunction reluActivation(relu, reluDerivative);

//leaky relu
float leakyRelu(float x){return (x > 0) ? x : 0.01f * x;}
float leakyReluDerivative(float x){return (x > 0) ? 1.0f : 0.01f;}
ActivationFunction leakyReluActivation(leakyRelu, leakyReluDerivative);

//softmax
std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> output(x.size());
    if (x.empty()) return output;
    float sum = 0.0f;
    for(int i = 0; i < x.size(); ++i) {
        output[i] = std::exp(x[i]);
        sum += output[i];
    }
    for(int i = 0; i <x.size(); ++i) {
        output[i] /= sum + 0.0000000001f; //small epsilon to avoid zero division errors
    }
    return output;
}

//mean squared error
float meanSquaredError(const std::vector<float>& predicted, const std::vector<float>& actual) {
    float sum = 0.0f;
    for (size_t i = 0; i < predicted.size(); ++i) {
        float diff = predicted[i] - actual[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
}
std::vector<float> meanSquaredErrorDerivative(const std::vector<float>& predicted, const std::vector<float>& actual) {
    std::vector<float> derivative(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        derivative[i] = 2.0f * (predicted[i] - actual[i]) / predicted.size();
    }
    return derivative;
}
LossFunction mseLoss(meanSquaredError, meanSquaredErrorDerivative);

struct MLPLayer{
    int inputSize; int outputSize;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    ActivationFunction activationFunction;
    std::vector<float> lastInput;
    std::vector<float> lastPreActivation;
    std::vector<float> lastOutput;

    MLPLayer(int input, int output, ActivationFunction activation) : inputSize(input), outputSize(output), activationFunction(activation) {
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
        lastInput = input; 
        lastPreActivation.assign(outputSize, 0.0f); 
        lastOutput.assign(outputSize, 0.0f);
        for (int i = 0; i < outputSize; ++i) {
            float activation = biases[i];
            for (int j = 0; j < inputSize; ++j) {
                activation += weights[i][j] * input[j];
            }
            lastPreActivation[i] = activation;
            lastOutput[i] = activationFunction.compute(activation);
        }
        return lastOutput;
    }
};

struct MLP{
    std::vector<MLPLayer> layers;
    LossFunction loss;
    ActivationFunction activation;
    bool softmaxOutput = false;

    MLP(std::vector<int> layerSizes, ActivationFunction activationFunction = reluActivation, LossFunction lossFunction = mseLoss, bool useSoftmaxOutput = false) : activation(activationFunction), loss(lossFunction), softmaxOutput(useSoftmaxOutput) {
        for (int i = 0; i < layerSizes.size() - 1; ++i) {
            layers.push_back(MLPLayer(layerSizes[i], layerSizes[i + 1], activation));
        }
    }

    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> output = input;
        for (MLPLayer& layer : layers) {
            output = layer.forward(output);
        }
        if (softmaxOutput) { output = softmax(output); }
        return output;
    }
};

int main() {
    MLP net({3, 5, 3}, leakyReluActivation, mseLoss, true);
    std::vector<float> result = net.forward({1.0f, 2.0f, 3.0f});
    for (float value : result) {
        std::cout << value << ", ";
    }
    return 0;
}