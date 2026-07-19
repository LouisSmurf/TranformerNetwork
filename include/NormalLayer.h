#ifndef NORMALLAYER_H
#define NORMALLAYER_H

#include <Eigen/Dense>
#include <random>
#include <cmath>

struct NormalLayer {
    int tokenSize;
    int inputSize;

    NormalLayer(int tokenSize, int embedDim);

    //learnable weights
    Eigen::RowVectorXf gamma;
    Eigen::RowVectorXf beta;

    Eigen::RowVectorXf gammaGradient;
    Eigen::RowVectorXf betaGradient;

    Eigen::MatrixXf lastInput;
    Eigen::MatrixXf lastXMean;
    Eigen::VectorXf lastMean;
    Eigen::VectorXf lastVar;
    Eigen::VectorXf lastStdDev;
    
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);
    Eigen::MatrixXf backward(const Eigen::MatrixXf& outputGradient, float learningRate);

    void initialiseParams();
};

#endif // NORMALLAYER_H