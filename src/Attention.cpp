#include "attention.h"


Attention::Attention(int inputSize, int tokenSize) : inputSize(inputSize), tokenSize(tokenSize) {
    input.resize(tokenSize, inputSize);
    query.resize(tokenSize, inputSize);
    key.resize(tokenSize, inputSize);
    value.resize(tokenSize, inputSize);
    output.resize(tokenSize, inputSize);

    queryWeights.resize(inputSize, inputSize);
    keyWeights.resize(inputSize, inputSize);
    valueWeights.resize(inputSize, inputSize);

    attentionWeights.resize(tokenSize, tokenSize);
}


Eigen::MatrixXf Attention::forward(const Eigen::MatrixXf& x) {
    input = x;
    query = input * queryWeights;
    key = input * keyWeights;
    value = input * valueWeights;

    float scale = 1.0f / std::sqrt(static_cast<float>(inputSize));
    Eigen::MatrixXf scores = query * key.transpose() * scale;

    attentionWeights = scores;
    for (int i = 0; i < tokenSize; ++i) {
        float max = attentionWeights.row(i).maxCoeff();
        attentionWeights.row(i).sum();
        attentionWeights.row(i) /= max;
    }

    output = attentionWeights * value;
    return output;
}