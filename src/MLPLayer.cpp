#include "MLPLayer.h"


struct MLPLayer{
    int inputSize; int outputSize;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    ActivationFunction activationFunction;
    std::vector<float> lastInput;
    std::vector<float> lastPreActivation;
    std::vector<float> lastOutput;

    MLPLayer(int input, int output, ActivationFunction activation) : inputSize(input), outputSize(output), activationFunction(activation) {
        weights.resize(outputSize, std::vector<float>(inputSize));
        biases.resize(outputSize);
        
        initialiseWeights();
    }

    void initialiseWeights() {
        std::random_device random;
        std::mt19937 generateNumber(random());
        std::normal_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] = dist(generateNumber);
            }
            biases[i] = dist(generateNumber);
        }
    }

    std::vector<float> forward(const std::vector<float>& input) {
        lastInput = input; 
        lastPreActivation.assign(outputSize, 0.0f); 
        lastOutput.assign(outputSize, 0.0f);
        for (int i = 0; i < outputSize; ++i) {
            float activation = biases[i];
            for (int j = 0; j < inputSize; ++j) {
                activation += weights[i][j] * input[j];
            }
            lastPreActivation[i] = activation;
            lastOutput[i] = activationFunction.compute(activation);
        }
        return lastOutput;
    }

    std::vector<float> backward(const std::vector<float>& outputGradient, float learningRate) {
        std::vector<float> inputGradient(inputSize, 0.0f);
        std::vector<float> preActivationGradient(outputSize, 0.0f);
        for (int i = 0; i < outputSize; ++i) {
            float delta = outputGradient[i] * activationFunction.computeDerivative(lastPreActivation[i]);
            preActivationGradient[i] = delta;
            for (int j = 0; j < inputSize; ++j) {
                inputGradient[j] += weights[i][j] * delta;
                weights[i][j] -= learningRate * delta * lastInput[j];
            }
            biases[i] -= learningRate * delta;
        }
        return inputGradient;
    }
};