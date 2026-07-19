#include "NormalLayer.h"



const float EPSILON = 1e-9f;

NormalLayer::NormalLayer(int tokenSize, int inputSize) : tokenSize(tokenSize), inputSize(inputSize) {
    gamma.resize(inputSize);
    beta.resize(inputSize);
    gammaGradient.resize(inputSize);
    betaGradient.resize(inputSize);

    initialiseParams();
}

void NormalLayer::initialiseParams() {
    gamma.setConstant(1.0f);
    beta.setZero();
}

Eigen::MatrixXf centered_multiply(const Eigen::VectorXf& vec, const Eigen::MatrixXf& mat) {
    return mat.array().colwise() * vec.array();
}

Eigen::MatrixXf NormalLayer::forward(const Eigen::MatrixXf& x) {
    lastInput = x;
    lastMean = lastInput.rowwise().mean();

    Eigen::MatrixXf centered = lastInput.colwise() - lastMean;
    lastVar = centered.array().square().rowwise().mean();
    lastStdDev = (lastVar.array() + EPSILON).sqrt();
    
    lastXMean = centered.array().colwise() / lastStdDev.array();
    Eigen::MatrixXf output = (lastXMean.array().rowwise() * gamma.array()).rowwise() + beta.array();
    
    return output;
}

Eigen::MatrixXf NormalLayer::backward(const Eigen::MatrixXf& outputGradient, float learningRate) {
    gammaGradient = (outputGradient.array() * lastXMean.array()).colwise().sum();
    betaGradient = outputGradient.colwise().sum();

    float N = static_cast<float>(inputSize);
    Eigen::MatrixXf inputGradient(tokenSize, inputSize);
    Eigen::MatrixXf dyGamma = outputGradient * gamma.asDiagonal();

    for (int i = 0; i < tokenSize; ++i) {
        float dyGammaSum = dyGamma.row(i).sum();
        float dyGammaXMeanSum = (dyGamma.row(i).array() * lastXMean.row(i).array()).sum();
        inputGradient.row(i) = (1.0f / lastStdDev(i)) * (dyGamma.row(i).array() - (dyGammaSum / N) - (lastXMean.row(i).array() * dyGammaXMeanSum / N));
    }

    gamma -= learningRate * gammaGradient;
    beta -= learningRate * betaGradient;

    return inputGradient;
}