#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <Eigen/Dense>
#include <functional>

struct ActivationFunction{
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> compute;
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> computeDerivative;
    ActivationFunction(std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> f, std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> df) : compute(f), computeDerivative(df) {}
};

struct LossFunction{
    std::function<float(const Eigen::MatrixXf&, const Eigen::MatrixXf&)> compute;
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, const Eigen::MatrixXf&)> computeDerivative;
    LossFunction(std::function<float(const Eigen::MatrixXf&, const Eigen::MatrixXf&)> f, std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, const Eigen::MatrixXf&)> df) : compute(f), computeDerivative(df) {}
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