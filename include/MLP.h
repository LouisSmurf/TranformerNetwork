#ifndef MLP_H
#define MLP_H


#include <Eigen/Dense>
#include <vector>

#include "MLPLayer.h"
#include "HelperFunctions.h"


struct MLP{
    std::vector<MLPLayer> layers;
    LossFunction loss;
    ActivationFunction activation;
    bool softmaxOutput = false;

    MLP(std::vector<int> layerSizes, ActivationFunction activationFunction = ActivationFunctions::ReLU, LossFunction lossFunction = LossFunctions::mseLoss, bool useSoftmaxOutput = false);

    Eigen::VectorXf forward(const Eigen::VectorXf& input);

    float train(const Eigen::VectorXf& input, const Eigen::VectorXf& target, float learningRate);
};


#endif