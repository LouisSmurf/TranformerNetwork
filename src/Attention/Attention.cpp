#include "Attention/Attention.h"


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

    initialiseWeights();
}


Eigen::MatrixXf Attention::forward(const Eigen::MatrixXf& x) {
    input = x;
    query = input * queryWeights;
    key = input * keyWeights;
    value = input * valueWeights;

    int currentSeqLen = x.rows(); 
    float scale = 1.0f / std::sqrt(static_cast<float>(inputSize));
    Eigen::MatrixXf scores = query * key.transpose() * scale;

    for (int i = 0; i < currentSeqLen; ++i) {
        for (int j = i + 1; j < currentSeqLen; ++j) {
            scores(i, j) = -std::numeric_limits<float>::infinity();
        }
    }

    attentionWeights = scores;
    
    for (int i = 0; i < currentSeqLen; ++i) {
        float max = attentionWeights.row(i).maxCoeff();
        attentionWeights.row(i) = (attentionWeights.row(i).array() - max).exp();
        float rowSum = attentionWeights.row(i).sum();
        attentionWeights.row(i) /= (rowSum + 1e-9f);
    }

    output = attentionWeights * value;
    return output;
}

Eigen::MatrixXf Attention::backward(const Eigen::MatrixXf& outputGradient, float learningRate) {
    int currentSeqLen = outputGradient.rows();
    Eigen::MatrixXf valueGradient = attentionWeights.transpose() * outputGradient;
    Eigen::MatrixXf attentionGradient = outputGradient * value.transpose();
    Eigen::MatrixXf scoresGradient = Eigen::MatrixXf::Zero(currentSeqLen, currentSeqLen);

    for (int i = 0; i < currentSeqLen; ++i) {
        Eigen::VectorXf attentionRow = attentionWeights.row(i).cwiseMax(1e-9f);
        Eigen::VectorXf attentionGradientRow = attentionGradient.row(i);
        float dotProduct = attentionRow.dot(attentionGradientRow);
        scoresGradient.row(i) = attentionRow.array() * (attentionGradientRow.array() - dotProduct);
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(inputSize));
    Eigen::MatrixXf rawScoresGradient = scoresGradient * scale;
    Eigen::MatrixXf queryGradient = rawScoresGradient * key;
    Eigen::MatrixXf keyGradient = rawScoresGradient.transpose() * query;

    Eigen::MatrixXf qWeightsGradient = input.transpose() * queryGradient;
    Eigen::MatrixXf kWeightsGradient = input.transpose() * keyGradient;
    Eigen::MatrixXf vWeightsGradient = input.transpose() * valueGradient;
    
    Eigen::MatrixXf inputGradient = (queryGradient * queryWeights.transpose()) + 
                                    (keyGradient * keyWeights.transpose()) + 
                                    (valueGradient * valueWeights.transpose());

    queryWeights -= learningRate * qWeightsGradient;
    keyWeights -= learningRate * kWeightsGradient;
    valueWeights -= learningRate * vWeightsGradient;

    return inputGradient;
}

void Attention::initialiseWeights() {
    std::random_device random;
    std::mt19937 generateNumber(random());
    
    float stddev = std::sqrt(2.0f / inputSize);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            queryWeights(i, j) = dist(generateNumber);
            keyWeights(i, j) = dist(generateNumber);
            valueWeights(i, j) = dist(generateNumber);
        }
    }
}