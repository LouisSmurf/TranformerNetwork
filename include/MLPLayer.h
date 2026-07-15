#ifndef MLP_LAYER_CPP
#define MLP_LAYER_CPP

#include <random>
#include "HelperFunctions.h"
#include <Eigen/Dense>

struct MLPLayer {
    int inputSize;
    int outputSize;
    
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    
    ActivationFunction activationFunction;
    
    Eigen::VectorXf lastInput;
    Eigen::VectorXf lastPreActivation;
    Eigen::VectorXf lastOutput;

    // Constructor
    MLPLayer(int input, int output, ActivationFunction activation);

    // Member Functions
    void initialiseWeights();
    
    Eigen::VectorXf forward(const Eigen::VectorXf& input);
    
    Eigen::VectorXf backward(const Eigen::VectorXf& outputGradient, float learningRate);
};

#endif