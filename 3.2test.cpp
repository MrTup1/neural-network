#include <iostream>
#include <vector>
#include <stdexcept>

// Include your two libraries
#include "matrix.hpp"
#include "neuralNetwork.hpp"

/**
 * @file main.cpp
 * @brief A test program for Section 3.2 of the lab.
 *
 * This program verifies that the NeuralNetwork class can be
 * constructed and that the feedForward function works as expected.
 *
 */
int main() {
    std::cout << "--- NeuralNetwork Class Test Program ---" << std::endl << std::endl;

    try {
        // --- 1. Test Constructor ---
        // As per Section 3.4, we're building a 4-bit decoder.
        // Input = 4 nodes, Output = 16 nodes.
        // Let's add one hidden layer of 8 nodes.
        std::vector<int> topology = {4, 8, 16};
        
        std::cout << "1. Creating a {4, 8, 16} network..." << std::endl;
        NeuralNetwork nn(topology, 0.1); // Learning rate 0.1
        nn.print(); // Use our utility function to check
        std::cout << "   ...Network created successfully." << std::endl << std::endl;


        // --- 2. Test feedForward ---
        std::cout << "2. Testing feedForward function..." << std::endl;
        
        // Create a sample 4-bit input (e.g., 1010)
        std::vector<double> input_vec = {1.0, 0.0, 1.0, 0.0};
        Matrix input = Matrix::fromVector(input_vec); // Convert to 4x1 Matrix
        
        std::cout << "Input (4x1):" << std::endl;
        input.print();

        // Feed the input through the network
        Matrix output = nn.feedForward(input);

        std::cout << "Output (16x1):" << std::endl;
        output.print();

        // --- 3. Verify Output Dimensions ---
        if (output.getRows() == 16 && output.getCols() == 1) {
            std::cout << "   ...Output dimensions are correct (16x1)!" << std::endl << std::endl;
        } else {
            std::cout << "   *** ERROR: Output dimensions are (" 
                      << output.getRows() << "x" << output.getCols() 
                      << "), expected (16x1)." << std::endl << std::endl;
        }

        // --- 4. Test Error Handling ---
        std::cout << "3. Testing error handling with bad input (2x1)..." << std::endl;
        try {
            std::vector<double> bad_input_vec = {1.0, 0.0};
            Matrix bad_input = Matrix::fromVector(bad_input_vec);
            nn.feedForward(bad_input);
            std::cout << "   *** ERROR: Network did not throw error on bad input." << std::endl;
        } catch (const std::exception& e) {
            std::cout << "   ...Caught expected error: " << e.what() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << std::endl << "--- Test Complete ---" << std::endl;
    return 0;
}