#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <vector>
#include <functional>
#include <algorithm>

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

struct ActivationFunctions{
    static ActivationFunction ReLU;
    static ActivationFunction leakyReLU;
};

struct LossFunctions{
    static LossFunction mseLoss;
    static LossFunction cceLoss;
};


#endif