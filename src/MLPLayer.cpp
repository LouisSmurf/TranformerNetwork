#include "MLPLayer.h"
#include <iostream>


MLPLayer::MLPLayer(int input, int output, ActivationFunction activation) : inputSize(input), outputSize(output), activationFunction(activation) {
    weights.resize(outputSize, inputSize);
    biases.resize(outputSize);
    
    initialiseWeights();
}

void MLPLayer::initialiseWeights() {
    std::random_device random;
    std::mt19937 generateNumber(random());
    
    float stddev = std::sqrt(2.0f / inputSize);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights(i, j) = dist(generateNumber);
        }
        biases(i) = 0.0f;
    }
}

Eigen::VectorXf MLPLayer::forward(const Eigen::VectorXf& input) {
    lastInput = input; 
    lastPreActivation = (weights * input) + biases;
    lastOutput = activationFunction.compute(lastPreActivation);
    return lastOutput;
}

Eigen::VectorXf MLPLayer::backward(const Eigen::VectorXf& outputGradient, float learningRate) {
    Eigen::VectorXf delta(outputSize);
    delta = outputGradient.cwiseProduct(activationFunction.computeDerivative(lastPreActivation));

    Eigen::VectorXf inputGradient = weights.transpose() * delta;

    weights -= learningRate * delta * lastInput.transpose();
    biases -= learningRate * delta;

    return inputGradient;
}