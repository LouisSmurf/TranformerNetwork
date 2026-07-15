#include "HelperFunctions.h"



const float EPSILON = 1e-9f;

struct ActivationFunction{
    std::function<float(float)> compute;
    std::function<float(float)> computeDerivative;
    ActivationFunction(std::function<float(float)> f, std::function<float(float)> df) : compute(f), computeDerivative(df) {}
};

struct LossFunction{
    std::function<float(const std::vector<float>&, const std::vector<float>&)> compute;
    std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)> computeDerivative;
    LossFunction(std::function<float(const std::vector<float>&, const std::vector<float>&)> f, std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)> df) : compute(f), computeDerivative(df) {}
};



//relu
float relu(float x){return std::max(0.0f, x);}
float reluDerivative(float x){return (x > 0) ? 1.0f : 0.0f;}
ActivationFunction reluActivation(relu, reluDerivative);

//leaky relu
float leakyRelu(float x){return (x > 0) ? x : 0.01f * x;}
float leakyReluDerivative(float x){return (x > 0) ? 1.0f : 0.01f;}
ActivationFunction leakyReluActivation(leakyRelu, leakyReluDerivative);

ActivationFunction ActivationFunctions::ReLU = reluActivation;
ActivationFunction ActivationFunctions::leakyReLU = leakyReluActivation;



//mean squared error
float meanSquaredError(const std::vector<float>& predicted, const std::vector<float>& actual) {
    float sum = 0.0f;
    for (size_t i = 0; i < predicted.size(); ++i) {
        float diff = predicted[i] - actual[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
}
std::vector<float> meanSquaredErrorDerivative(const std::vector<float>& predicted, const std::vector<float>& actual) {
    std::vector<float> derivative(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        derivative[i] = 2.0f * (predicted[i] - actual[i]) / predicted.size();
    }
    return derivative;
}
LossFunction mseLoss(meanSquaredError, meanSquaredErrorDerivative);

float categoricalCrossEntropy(const std::vector<float>& predicted, const std::vector<float>& actual) {
    float loss = 0.0f;
    for (size_t i = 0; i < predicted.size(); ++i) {
        if (actual[i] == 1.0f) {
            loss -= std::log(predicted[i] + EPSILON);
        }
    }
    return loss;
}

std::vector<float> categoricalCrossEntropyDerivative(const std::vector<float>& predicted, const std::vector<float>& actual) {
    std::vector<float> derivative(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        derivative[i] = predicted[i] - actual[i];
    }
    return derivative;
}

LossFunction cceLoss(categoricalCrossEntropy, categoricalCrossEntropyDerivative);


LossFunction LossFunctions::mseLoss = mseLoss;
LossFunction LossFunctions::cceLoss = cceLoss;