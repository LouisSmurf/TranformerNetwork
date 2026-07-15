#include "MLP.h"
#include <iostream>


const float EPSILON = 1e-9f;

//softmax(gemini generated)
Eigen::VectorXf softmax(const Eigen::VectorXf& x) {
    if (x.size() == 0) return Eigen::VectorXf(0);
    
    float max = x.maxCoeff();
    Eigen::VectorXf output = (x.array() - max).exp();
    float sum = output.sum();
    output /= sum + EPSILON; //small epsilon to avoid zero division errors

    std::cout << "returning output" << std::endl;
    return output;
}

Eigen::VectorXf softmaxDerivativeBackward(const Eigen::VectorXf& softmaxOutput, const Eigen::VectorXf& outputGradient) {
    Eigen::MatrixXf diagSoftmax = softmaxOutput.asDiagonal();
    Eigen::MatrixXf outerSoftmax = softmaxOutput * softmaxOutput.transpose();
    Eigen::MatrixXf jacobian = diagSoftmax - outerSoftmax;

    return jacobian * outputGradient;
}

//MLP
MLP::MLP(std::vector<int> layerSizes, ActivationFunction activationFunction, LossFunction lossFunction, bool useSoftmaxOutput) : activation(activationFunction), loss(lossFunction), softmaxOutput(useSoftmaxOutput) {
    for (int i = 0; i < layerSizes.size() - 1; ++i) {
        layers.push_back(MLPLayer(layerSizes[i], layerSizes[i + 1], activation));
    }
}

Eigen::VectorXf MLP::forward(const Eigen::VectorXf& input) {
    Eigen::VectorXf output = input;
    for (MLPLayer& layer : layers) {
        output = layer.forward(output);
    }

    if (softmaxOutput) { output = softmax(output); }
    return output;
}

float MLP::train(const Eigen::VectorXf& input, const Eigen::VectorXf& target, float learningRate) {
    Eigen::VectorXf predicted = forward(input);
    float lossValue = loss.compute(predicted, target);

    Eigen::VectorXf lossGradient = loss.computeDerivative(predicted, target);

    if (softmaxOutput) {
        lossGradient = softmaxDerivativeBackward(predicted, lossGradient);
    }

    for (int i = layers.size() - 1; i >= 0; --i) {
        lossGradient = layers[i].backward(lossGradient, learningRate);
    }

    return lossValue;
}