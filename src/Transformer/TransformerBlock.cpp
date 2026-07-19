#include "Transformer/TransformerBlock.h"


TransformerBlock::TransformerBlock(int inputSize, int tokenSize, MultiLayerMLP mlp, Attention attention) : inputSize(inputSize), tokenSize(tokenSize), mlp(mlp), attention(attention),normalLayer(tokenSize, inputSize),normalLayer2(tokenSize, inputSize) {
    lastInput.resize(tokenSize, inputSize);
    lastNormalLayerOutput.resize(tokenSize, inputSize);
    lastAttentionOutput.resize(tokenSize, inputSize);
    lastPostAttentionOutput.resize(tokenSize, inputSize);
    lastNormalLayer2Output.resize(tokenSize, inputSize);
    lastMlpOutput.resize(tokenSize, inputSize);
}


Eigen::MatrixXf TransformerBlock::forward(const Eigen::MatrixXf& input) {
    lastInput = input;
    lastNormalLayerOutput = normalLayer.forward(lastInput);
    lastAttentionOutput = attention.forward(lastNormalLayerOutput);
    lastPostAttentionOutput = lastInput + lastAttentionOutput;
    lastNormalLayer2Output = normalLayer2.forward(lastPostAttentionOutput);
    lastMlpOutput = mlp.forward(lastNormalLayer2Output);

    Eigen::MatrixXf finalOutput = lastPostAttentionOutput + lastMlpOutput;

    return finalOutput;
}



Eigen::MatrixXf TransformerBlock::backward(const Eigen::MatrixXf& outputGradient, float learningRate) {
    Eigen::MatrixXf clippedGradient = outputGradient.cwiseMin(1.0f).cwiseMax(-1.0f);
    Eigen::MatrixXf mlpOutputGradient = clippedGradient;
    Eigen::MatrixXf postAttentionGradientPath = clippedGradient;
    Eigen::MatrixXf normalLayer2OutputGradient = mlp.backward(mlpOutputGradient, learningRate);
    Eigen::MatrixXf postAttentionFromMlpGradient = normalLayer2.backward(normalLayer2OutputGradient, learningRate);
    Eigen::MatrixXf postAttentionGradient = postAttentionGradientPath + postAttentionFromMlpGradient;
    Eigen::MatrixXf attentionOutputGradient = postAttentionGradient;
    Eigen::MatrixXf inputPathGradient = postAttentionGradient;
    Eigen::MatrixXf normalLayerOutputGradient = attention.backward(attentionOutputGradient, learningRate);
    Eigen::MatrixXf inputFromAttentionGradient = normalLayer.backward(normalLayerOutputGradient, learningRate);
    Eigen::MatrixXf inputGradient = inputPathGradient + inputFromAttentionGradient;

    return inputGradient;
}