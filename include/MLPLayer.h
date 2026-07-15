#ifndef MLP_LAYER_CPP
#define MLP_LAYER_CPP

#include <vector>
#include <random>
#include "HelperFunctions.h"

struct MLPLayer {
    int inputSize;
    int outputSize;
    
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    
    ActivationFunction activationFunction;
    
    std::vector<float> lastInput;
    std::vector<float> lastPreActivation;
    std::vector<float> lastOutput;

    // Constructor
    MLPLayer(int input, int output, ActivationFunction activation);

    // Member Functions
    void initialiseWeights();
    
    std::vector<float> forward(const std::vector<float>& input);
    
    std::vector<float> backward(const std::vector<float>& outputGradient, float learningRate);
};

#endif