#include <iostream>
#include <vector>
#include <stdexcept>
#include <iomanip>  // For std::setprecision
#include <string>   // For std::string
#include <sstream>  // For std::stringstream
#include <cmath>    // For std::pow
#include <string>
#include <charconv> // from_char, to_char

// Include your two libraries
#include "matrix.hpp"
#include "neuralNetwork.hpp"

/**
 * @file main.cpp
 * @brief Section 3.4: Train the Neural Network
 *
 * This program implements the full training loop for the 4-bit
 * binary decoder task.
 */

// --- Helper Function to Get Network's "Guess" ---
/**
 * @brief Finds the index of the highest value in a 16x1 output matrix.
 * @param output The output matrix from the network.
 * @return The index (0-15) of the highest value.
 */
int getMaxIndex(const Matrix& m) {
    if (m.getCols() != 1) {
        throw std::invalid_argument("getMaxIndex expects a column vector.");
    }

    std::vector<double> v = m.toVector(); // Use your toVector function
    if (v.empty()) {
        return -1;
    }

    int max_index = 0;
    double max_value = v[0];

    for (int i = 1; i < v.size(); ++i) {
        if (v[i] > max_value) {
            max_value = v[i];
            max_index = i;
        }
    }
    return max_index;
}

// --- Helper Function to Format Binary Strings ---
/**
 * @brief Formats a 4-bit binary string for an integer.
 * @param n The integer (0-15).
 * @return A string (e.g., "0011" for n=3).
 */
std::string formatBinary(int n) {
    std::string s = "";
    for (int i = 3; i >= 0; --i) {
        s += ((n >> i) & 1) ? "1" : "0";
    }
    return s;
}


int main() {
    try {
        // --- 1. Set up Network ---
        std::vector<int> topology = {4, 10, 16}; // 4-in, 10-hidden, 16-out
        double learning_rate = 0.1;
        NeuralNetwork nn(topology, learning_rate);
        
        std::cout << "Created a {4, 10, 16} network." << std::endl;

        // --- 2. Generate Training Data (4-bit decoder) ---
        std::vector<Matrix> all_inputs;
        std::vector<Matrix> all_targets;

        for (int i = 0; i < 16; ++i) {
            // Create the 4x1 input vector
            std::vector<double> input_vec(4, 0.0);
            int n = i;
            input_vec[3] = n % 2; n /= 2; // LSB (2^0)
            input_vec[2] = n % 2; n /= 2; // (2^1)
            input_vec[1] = n % 2; n /= 2; // (2^2)
            input_vec[0] = n % 2;         // MSB (2^3)
            all_inputs.push_back(Matrix::fromVector(input_vec));

            // Create the 16x1 one-hot target vector
            std::vector<double> target_vec(16, 0.0);
            target_vec[i] = 1.0; // Set the correct output node to 1
            all_targets.push_back(Matrix::fromVector(target_vec));
        }
        std::cout << "Generated 16 input/target pairs." << std::endl;


        // --- 3. Run the Training Loop ---
        int epochs = 20000;
        std::cout << "Starting training for " << epochs << " epochs..." << std::endl;

        for (int ep = 0; ep < epochs; ++ep) {
            double epoch_loss = 0.0;
            
            // Train on all 16 data points in each epoch
            for (int i = 0; i < 16; ++i) {
                // 1. Feed the input forward
                nn.feedForward(all_inputs[i]);
                
                // 2. Update the weights (backpropagation)
                //    and get the loss for this sample
                epoch_loss += nn.update(all_targets[i]);
            }

            // Print the average loss for this epoch (just like the screenshot)
            if (ep % 1000 == 0 || ep == epochs - 1) {
                std::cout << std::fixed << std::setprecision(10)
                          << "EPOCH " << std::setw(5) << ep
                          << ", avg_loss = " << (epoch_loss / 16.0)
                          << std::endl;
            }
        }
        std::cout << "Training complete." << std::endl << std::endl;


        // --- 4. Test the Trained Network ---
        std::cout << "--- Testing Network ---" << std::endl;
        
        std::string line;
        while (true) {
            std::cout << "Enter a decimal number to test:";
            std::getline(std::cin, line);
            int integerInput = 0;

            if (line == "exit" || line == "q") {
                break;
            }

            try {
                integerInput = std::stoi(line);
                if (integerInput > 15 || integerInput < 0) {
                    std::cout << std::endl;
                    std::cout << "WRONG! Number must be within 0 - 15!" << std::endl;
                    std::cout << std::endl;
                    continue;
                }
                std::cout << "Success! Your number is: " << integerInput << std::endl;
                
            } 
            // 3. Catch any errors
            catch (const std::invalid_argument& e) {
                std::cout << std::endl;
                std::cout << "Error: That's not a valid number." << std::endl;
                std::cout << std::endl;
                continue;
            }

            Matrix input = all_inputs[integerInput];
            Matrix output = nn.feedForward(input);
            int guess = getMaxIndex(output);

            std::cout << "Input: " << formatBinary(integerInput) 
                      << " (Decimal: " << std::setw(2) << integerInput << ")" << std::endl;
            std::cout << "Guess: " << std::setw(2) << guess << std::endl;
            std::cout << "Output:" <<std::endl;
            output.print();
        }


    } catch (const std::exception& e) {
        std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}