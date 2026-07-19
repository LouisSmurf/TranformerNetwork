#include "MLP/MLPLayer.h"


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

Eigen::MatrixXf MLPLayer::forward(const Eigen::MatrixXf& input) {
    lastInput = input;  //[tokensSize, inputSize]
    lastPreActivation = (input * weights.transpose()).rowwise() + biases.transpose();
    lastOutput = activationFunction.compute(lastPreActivation);
    return lastOutput; // [tokenSize, outputSize]
}

Eigen::MatrixXf MLPLayer::backward(const Eigen::MatrixXf& outputGradient, float learningRate) {
    //[tokenSize, outputSize]
    Eigen::MatrixXf delta = outputGradient.cwiseProduct(activationFunction.computeDerivative(lastPreActivation));

    Eigen::MatrixXf inputGradient = delta * weights; // [tokenSize, inputSize]
    Eigen::MatrixXf dWeights = delta.transpose() * lastInput; //[outputSize, inputSize]
    Eigen::VectorXf dBiases = delta.colwise().sum().transpose();

    weights -= learningRate * dWeights;
    biases -= learningRate * dBiases;

    return inputGradient;
}