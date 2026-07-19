#include "MLP/MultiLayerMLP.h"

MultiLayerMLP::MultiLayerMLP(const std::vector<int>& layerSizes, ActivationFunction activation) {
    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
        layers.emplace_back(layerSizes[i], layerSizes[i+1], activation);
    }
    intermediateActivations.resize(layers.size());
}


Eigen::MatrixXf MultiLayerMLP::forward(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf current = input;
    for (size_t i = 0; i < layers.size(); ++i) {
        current = layers[i].forward(current);
        intermediateActivations[i] = current;
    }
    return current;
}

Eigen::MatrixXf MultiLayerMLP::backward(const Eigen::MatrixXf& outputGradient, float learningRate) {
    Eigen::MatrixXf currentGradient = outputGradient;
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        currentGradient = layers[i].backward(currentGradient, learningRate);
    }
    return currentGradient;
}