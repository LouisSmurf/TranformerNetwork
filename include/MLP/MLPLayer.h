#ifndef MLP_LAYER_CPP
#define MLP_LAYER_CPP

#include <random>
#include "HelperFunctions.h"
#include <Eigen/Dense>
#include <iostream>

struct MLPLayer {
    int inputSize;
    int outputSize;
    
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    
    ActivationFunction activationFunction;
    
    Eigen::MatrixXf lastInput;
    Eigen::MatrixXf lastPreActivation;
    Eigen::MatrixXf lastOutput;

    MLPLayer(int input, int output, ActivationFunction activation);

    void initialiseWeights();
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    Eigen::MatrixXf backward(const Eigen::MatrixXf& outputGradient, float learningRate);
};

#endif