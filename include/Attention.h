#ifndef ATTENTION_H
#define ATTENTION_H

#include <Eigen/Dense>

struct Attention {
    int inputSize;
    int tokenSize;

    // [tokenSize, inputSize]
    Eigen::MatrixXf input;
    Eigen::MatrixXf query;
    Eigen::MatrixXf key;
    Eigen::MatrixXf value;
    Eigen::MatrixXf output;

    // learnt weights [inputSize, inputSize]
    Eigen::MatrixXf queryWeights;
    Eigen::MatrixXf keyWeights;
    Eigen::MatrixXf valueWeights;

    // [tokenSize, TokenSize]
    Eigen::MatrixXf attentionWeights;

    Attention(int inputSize, int tokenSize);
    void initialiseWeights();

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);
    Eigen::MatrixXf backward(const Eigen::MatrixXf& outputGradient, float learningRate);
};


#endif // ATTENTION_H