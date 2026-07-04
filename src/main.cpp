#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <string>
#include <map>
#include <iterator>
#include <algorithm>
#include <fstream>

std::vector<char> tokens = {'*', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
std::map<char, int> charToIndex;
struct TrainingExample { std::vector<float> input; std::vector<float> target; };


std::vector<float> encodeSequence(const std::string& seq) {
    std::vector<float> input(4 * 27, 0.0f); // 4 tokens * 27 size
    for (int i = 0; i < 4; ++i) {
        if (i < seq.size()) {
            int charIdx = charToIndex[seq[i]];
            input[i * 27 + charIdx] = 1.0f;
        } else {
            // Treat missing tokens as index 0 ('*')
            input[i * 27 + 0] = 1.0f; 
        }
    }
    return input;
}



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
        output[i] /= sum + 0.0000000001f; //small epsilon to avoid zero division errors
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



const float EPSILON = 1e-9f;

float categoricalCrossEntropy(const std::vector<float>& predicted, const std::vector<float>& actual) {
    float loss = 0.0f;
    for (size_t i = 0; i < predicted.size(); ++i) {
        // Only consider the 'actual' class (where target is 1)
        if (actual[i] == 1.0f) {
            loss -= std::log(predicted[i] + EPSILON);
        }
    }
    return loss;
}

std::vector<float> categoricalCrossEntropyDerivative(const std::vector<float>& predicted, const std::vector<float>& actual) {
    std::vector<float> derivative(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        // When Softmax is combined with Cross-Entropy, 
        // the derivative simplifies to: predicted - actual
        derivative[i] = predicted[i] - actual[i];
    }
    return derivative;
}

LossFunction cceLoss(categoricalCrossEntropy, categoricalCrossEntropyDerivative);

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

struct MLP{
    std::vector<MLPLayer> layers;
    LossFunction loss;
    ActivationFunction activation;
    bool softmaxOutput = false;

    MLP(std::vector<int> layerSizes, ActivationFunction activationFunction = reluActivation, LossFunction lossFunction = mseLoss, bool useSoftmaxOutput = false) : activation(activationFunction), loss(lossFunction), softmaxOutput(useSoftmaxOutput) {
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
    
        // REMOVE the line: if (softmaxOutput) { lossGradient = softmaxDerivativeBackward(predicted, lossGradient); }
        // Because CCE + Softmax derivative is already calculated in lossGradient.

        for (int i = layers.size() - 1; i >= 0; --i) {
            lossGradient = layers[i].backward(lossGradient, learningRate);
        }
        return lossValue;
    }
};

std::vector<float> toOneHot(int index, int alphabetSize = 27) {
    std::vector<float> vec(alphabetSize, 0.0f);
    vec[index] = 1.0f;
    return vec;
}

std::string oneHotToString(const std::vector<float>& oneHot) {
    std::string result = "";
    int numTokens = 4;
    int alphabetSize = 27;

    for (int i = 0; i < numTokens; ++i) {
        // Calculate the start and end iterators for the current chunk of 27
        auto start = oneHot.begin() + (i * alphabetSize);
        auto end = start + alphabetSize;

        // Find the index of the element with the highest probability/value
        auto maxIt = std::max_element(start, end);
        int idx = std::distance(start, maxIt);

        // Map index back to the character
        if (idx >= 0 && idx < tokens.size()) {
            result += tokens[idx];
        } else {
            result += '?'; // Error fallback
        }
    }
    return result;
}

void output(TrainingExample ex){
    std::cout<<"INPUT: [";
    for(int i = 0; i<ex.input.size(); ++i){
        std::cout<<ex.input[i]<<", ";
    }
    std::cout<<"]\n";
    std::cout<<oneHotToString(ex.input)<<"\n\n";

    std::cout<<"OUTPUT: [";
    for(int i = 0; i<ex.target.size(); ++i){
        std::cout<<ex.target[i]<<", ";
    }
    std::cout<<"]\n";
}

std::vector<std::string> loadWords(const std::string& dir) {
    std::vector<std::string> words;

    std::ifstream file(dir);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << dir << '\n';
        return words;
    }

    std::string word;
    while (std::getline(file, word)) {
        if (!word.empty()) {
            words.push_back(word);
        }
    }

    return words;
}


int EPOCHS = 2500;
float LEARNINGRATE = 0.0005f;
int main() {
    for (int i = 0; i < tokens.size(); ++i) {
        charToIndex[tokens[i]] = i;
    }
    std::vector<TrainingExample> dataset;
    std::vector<std::string> words = loadWords("words.txt");
    for (const std::string& word : words) {
        for (int i = 0; i < 4; ++i) {
            // Get the current sequence "wal"
            std::string sub = word.substr(0, i + 1);
            // Pad with '*' if the window is smaller than 4: "***w", "**wa", "*wal"
            while (sub.size() < 4) sub = '*' + sub; 
            
            TrainingExample ex;
            ex.input = encodeSequence(sub); // This creates the 108-size vector
            
            // The target is the next character
            ex.target = toOneHot(charToIndex[word[i + 1]], 27); 
            dataset.push_back(ex);
            //output(ex);
        }
    }


    MLP net({108, 128, 64, 27}, leakyReluActivation, cceLoss, true);
    std::cout << "Starting training..." << std::endl;
    for(int epoch = 0; epoch < EPOCHS; ++epoch) {
        float totalLoss = 0.0f;
        for (const auto& example : dataset) {
            totalLoss += net.train(example.input, example.target, LEARNINGRATE);
        }
        
        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << " | Avg Loss: " << totalLoss / dataset.size() << std::endl;
        }
    }

    std::cout << "Training complete!" << std::endl;

    //data
    std::string sub = "gr";
    while (sub.size() < 4) sub = '*' + sub; 
    std::vector<float> input = encodeSequence(sub);
    std::cout << oneHotToString(input) << "\n";

    std::vector<float> output = net.forward(input);
    for(int i=0; i<27; ++i){
        std::cout<<output[i]<<", ";
    }
    return 0;
}