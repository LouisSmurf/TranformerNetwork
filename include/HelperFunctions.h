#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <Eigen/Dense>
#include <functional>

struct ActivationFunction{
    std::function<Eigen::VectorXf(const Eigen::VectorXf&)> compute;
    std::function<Eigen::VectorXf(const Eigen::VectorXf&)> computeDerivative;
    ActivationFunction(std::function<Eigen::VectorXf(const Eigen::VectorXf&)> f, std::function<Eigen::VectorXf(const Eigen::VectorXf&)> df) : compute(f), computeDerivative(df) {}
};

struct LossFunction{
    std::function<float(const Eigen::VectorXf&, const Eigen::VectorXf&)> compute;
    std::function<Eigen::VectorXf(const Eigen::VectorXf&, const Eigen::VectorXf&)> computeDerivative;
    LossFunction(std::function<float(const Eigen::VectorXf&, const Eigen::VectorXf&)> f, std::function<Eigen::VectorXf(const Eigen::VectorXf&, const Eigen::VectorXf&)> df) : compute(f), computeDerivative(df) {}
};

struct ActivationFunctions{
    static ActivationFunction ReLU;
    static ActivationFunction leakyReLU;
    static ActivationFunction sigmoid;
};

struct LossFunctions{
    static LossFunction mseLoss;
    static LossFunction cceLoss;
};


#endif