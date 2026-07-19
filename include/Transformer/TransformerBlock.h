#ifndef TRANSFORMERBLOCK_H
#define TRANSFORMERBLOCK_H

#include <Eigen/Dense>
#include "Attention/Attention.h"
#include "MLP/MultiLayerMLP.h"
#include "NormalLayer.h"

struct TransformerBlock {
    int inputSize;
    int tokenSize;

    TransformerBlock(int inputSize, int tokenSize, MultiLayerMLP mlp, Attention attention);

    //layers
    MultiLayerMLP mlp;
    NormalLayer normalLayer;
    Attention attention;
    NormalLayer normalLayer2;

    //caches
    Eigen::MatrixXf lastInput;
    Eigen::MatrixXf lastNormalLayerOutput;
    Eigen::MatrixXf lastAttentionOutput;
    Eigen::MatrixXf lastPostAttentionOutput;
    Eigen::MatrixXf lastNormalLayer2Output;
    Eigen::MatrixXf lastMlpOutput;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    Eigen::MatrixXf backward(const Eigen::MatrixXf& outputGradient, float learningRate);
};

#endif // TRANSFORMERBLOCK_H