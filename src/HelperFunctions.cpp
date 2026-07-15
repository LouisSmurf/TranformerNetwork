#include "HelperFunctions.h"



const float EPSILON = 1e-9f;


//relu
Eigen::VectorXf relu(Eigen::VectorXf x){return x.cwiseMax(0.0f);}
Eigen::VectorXf reluDerivative(Eigen::VectorXf x){return (x.array() > 0.0f).select(Eigen::VectorXf::Ones(x.size()), Eigen::VectorXf::Zero(x.size()));}
ActivationFunction reluActivation(relu, reluDerivative);

//leaky relu
Eigen::VectorXf leakyRelu(Eigen::VectorXf x){return (x.array() > 0.0f).select(x, 0.01f * x);}
Eigen::VectorXf leakyReluDerivative(Eigen::VectorXf x){return (x.array() > 0.0f).select(Eigen::VectorXf::Ones(x.size()), Eigen::VectorXf::Constant(x.size(), 0.01f));}
ActivationFunction leakyReluActivation(leakyRelu, leakyReluDerivative);

//sigmoid
Eigen::VectorXf sigmoid(Eigen::VectorXf x){return 1.0f / (1.0f + (-x).array().exp());}
Eigen::VectorXf sigmoidDerivative(Eigen::VectorXf x){Eigen::VectorXf s = sigmoid(x); return s.array() * (1.0f - s.array());}
ActivationFunction sigmoidActivation(sigmoid, sigmoidDerivative);


ActivationFunction ActivationFunctions::ReLU = reluActivation;
ActivationFunction ActivationFunctions::leakyReLU = leakyReluActivation;
ActivationFunction ActivationFunctions::sigmoid = sigmoidActivation;



//mean squared error
float meanSquaredError(const Eigen::VectorXf& predicted, const Eigen::VectorXf& actual) {
    return (predicted - actual).squaredNorm() / static_cast<float>(predicted.size());
}
Eigen::VectorXf meanSquaredErrorDerivative(const Eigen::VectorXf& predicted, const Eigen::VectorXf& actual) {
    return 2.0f * (predicted - actual) / static_cast<float>(predicted.size());
}
LossFunction meanSquaredErrorLoss(meanSquaredError, meanSquaredErrorDerivative);



float categoricalCrossEntropy(const Eigen::VectorXf& predicted, const Eigen::VectorXf& actual) {
    return - (actual.array() * (predicted.array() + EPSILON).log()).sum();
}
Eigen::VectorXf categoricalCrossEntropyDerivative(const Eigen::VectorXf& predicted, const Eigen::VectorXf& actual) {
    return predicted - actual;
}
LossFunction categoricalCrossEntropyLoss(categoricalCrossEntropy, categoricalCrossEntropyDerivative);


LossFunction LossFunctions::mseLoss = meanSquaredErrorLoss;
LossFunction LossFunctions::cceLoss = categoricalCrossEntropyLoss;