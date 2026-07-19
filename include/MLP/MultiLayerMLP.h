#ifndef MULTILAYERMLP_H
#define MULTILAYERMLP_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>

#include "MLPLayer.h"
#include "HelperFunctions.h"

struct MultiLayerMLP {
    std::vector<MLPLayer> layers;
    std::vector<Eigen::MatrixXf> intermediateActivations;

    MultiLayerMLP(const std::vector<int>& layerSizes, ActivationFunction activation);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    Eigen::MatrixXf backward(const Eigen::MatrixXf& outputGradient, float learningRate);
};

#endif // MULTILAYERMLP_H