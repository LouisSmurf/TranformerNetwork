#include "MLP.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>



float EPOCHS = 1000;
float LEARNING_RATE = 0.01f;
int main() {
    std::vector<int> layerSizes = {2, 512, 512, 1};
    MLP mlp(layerSizes, ActivationFunctions::leakyReLU, LossFunctions::mseLoss, false);

    std::vector<Eigen::VectorXf> input = {
        (Eigen::VectorXf(2) << 1.0f, 1.0f).finished(),
        (Eigen::VectorXf(2) << 1.0f, 0.0f).finished(),
        (Eigen::VectorXf(2) << 0.0f, 1.0f).finished(),
        (Eigen::VectorXf(2) << 0.0f, 0.0f).finished()
    };

    std::vector<Eigen::VectorXf> target = {
        (Eigen::VectorXf(1) << 0.0f).finished(),
        (Eigen::VectorXf(1) << 1.0f).finished(),
        (Eigen::VectorXf(1) << 1.0f).finished(),
        (Eigen::VectorXf(1) << 0.0f).finished()
    };

    //Training
    float loss = 2.0f;
    std::cout << "Starting training...\n";
    for(int i = 0; i < EPOCHS; ++i) {
        std::cout << "------------------------------\n";
        std::cout << "Epoch " << i << "\n";
        for(int j = 0; j < input.size(); ++j) {
            loss = mlp.train(input[j], target[j], LEARNING_RATE);
        }
        std::cout << "Loss: " << loss << "\n";
        std::cout << "------------------------------\n";
    }
    
    std::cout << "Finished training!\n";
    //results
    for(int i = 0; i < input.size(); ++i) {
        std::cout << "Prediction for sample " << i << ": " << mlp.forward(input[i]).transpose() << std::endl;
        std::cout << "Target for sample " << i << ": " << target[i].transpose() << std::endl;
    }
}