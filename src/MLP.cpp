#include "MLP.h"

//softmax
std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> output(x.size());
    if (x.empty()) return output;
    float sum = 0.0f;
    for(int i = 0; i < x.size(); ++i) {
        output[i] = std::exp(x[i]);
        sum += output[i];
    }
    for(int i = 0; i <x.size(); ++i) {
        output[i] /= sum + EPSILON; //small epsilon to avoid zero division errors
    }
    return output;
}

std::vector<float> softmaxDerivativeBackward(const std::vector<float>& softmaxOutput, const std::vector<float>& outputGradient) {
    std::vector<float> inputGradient(softmaxOutput.size(), 0.0f);
    for (int i = 0; i < softmaxOutput.size(); ++i) {
        for (int j = 0; j < softmaxOutput.size(); ++j) {
            float derivative;
            if (i == j) {
                derivative = softmaxOutput[i] * (1.0f - softmaxOutput[i]);
            } 
            else {
                derivative = -softmaxOutput[i] * softmaxOutput[j];
            }
            inputGradient[i] += outputGradient[j] * derivative;
        }
    }
    return inputGradient;
}

//MLP
struct MLP{
    std::vector<MLPLayer> layers;
    LossFunction loss;
    ActivationFunction activation;
    bool softmaxOutput = false;

    MLP(std::vector<int> layerSizes, ActivationFunction activationFunction = ActivationFunctions::ReLU, LossFunction lossFunction = LossFunctions::mseLoss, bool useSoftmaxOutput = false) : activation(activationFunction), loss(lossFunction), softmaxOutput(useSoftmaxOutput) {
        for (int i = 0; i < layerSizes.size() - 1; ++i) {
            layers.push_back(MLPLayer(layerSizes[i], layerSizes[i + 1], activation));
        }
    }

    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> output = input;
        for (MLPLayer& layer : layers) {
            output = layer.forward(output);
        }
        if (softmaxOutput) { output = softmax(output); }
        return output;
    }

    float train(const std::vector<float>& input, const std::vector<float>& target, float learningRate) {
        std::vector<float> predicted = forward(input);
        float lossValue = loss.compute(predicted, target);
    
        // Get the gradient
        std::vector<float> lossGradient = loss.computeDerivative(predicted, target);

        for (int i = layers.size() - 1; i >= 0; --i) {
            lossGradient = layers[i].backward(lossGradient, learningRate);
        }
        return lossValue;
    }
};