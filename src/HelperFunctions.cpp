#include "HelperFunctions.h"



const float EPSILON = 1e-9f;


//relu
Eigen::MatrixXf relu(const Eigen::MatrixXf& x){return x.cwiseMax(0.0f);}
Eigen::MatrixXf reluDerivative(const Eigen::MatrixXf& x) {return (x.array() > 0.0f).select(Eigen::MatrixXf::Ones(x.rows(), x.cols()), Eigen::MatrixXf::Zero(x.rows(), x.cols()));}
ActivationFunction reluActivation(relu, reluDerivative);

//leaky relu
Eigen::MatrixXf leakyRelu(const Eigen::MatrixXf& x){return (x.array() > 0.0f).select(x, 0.01f * x);}
Eigen::MatrixXf leakyReluDerivative(const Eigen::MatrixXf& x) {return (x.array() > 0.0f).select(Eigen::MatrixXf::Ones(x.rows(), x.cols()),Eigen::MatrixXf::Constant(x.rows(), x.cols(), 0.01f));}
ActivationFunction leakyReluActivation(leakyRelu, leakyReluDerivative);

//sigmoid(cba to turn it into matrix)
// Eigen::VectorXf sigmoid(Eigen::VectorXf x){return 1.0f / (1.0f + (-x).array().exp());}
// Eigen::VectorXf sigmoidDerivative(Eigen::VectorXf x){Eigen::VectorXf s = sigmoid(x); return s.array() * (1.0f - s.array());}
// ActivationFunction sigmoidActivation(sigmoid, sigmoidDerivative);


ActivationFunction ActivationFunctions::ReLU = reluActivation;
ActivationFunction ActivationFunctions::leakyReLU = leakyReluActivation;
//ActivationFunction ActivationFunctions::sigmoid = sigmoidActivation; //same here



//mean squared error
float meanSquaredError(const Eigen::MatrixXf& predicted, const Eigen::MatrixXf& actual) {
    return (predicted - actual).squaredNorm() / static_cast<float>(predicted.size());
}
Eigen::MatrixXf meanSquaredErrorDerivative(const Eigen::MatrixXf& predicted, const Eigen::MatrixXf& actual) {
    return 2.0f * (predicted - actual) / static_cast<float>(predicted.size());
}
LossFunction meanSquaredErrorLoss(meanSquaredError, meanSquaredErrorDerivative);



float categoricalCrossEntropy(const Eigen::MatrixXf& predicted, const Eigen::MatrixXf& actual) {
    return - (actual.array() * (predicted.array() + EPSILON).log()).sum() / static_cast<float>(predicted.rows());
}
Eigen::MatrixXf categoricalCrossEntropyDerivative(const Eigen::MatrixXf& predicted, const Eigen::MatrixXf& actual) {
    return (predicted - actual) / static_cast<float>(predicted.rows());
}
LossFunction categoricalCrossEntropyLoss(categoricalCrossEntropy, categoricalCrossEntropyDerivative);


LossFunction LossFunctions::mseLoss = meanSquaredErrorLoss;
LossFunction LossFunctions::cceLoss = categoricalCrossEntropyLoss;