#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include "matrix.hpp"


class NeuralNetwork {
private:
    // --- Member Variables ---

    /**
     * @brief The network's topology, e.g., {4, 8, 16} for 4-in, 8-hidden, 16-out.
     */
    std::vector<int> layer_nodes;

    std::vector<std::string> layer_activations;

    /**
     * @brief A list of Weight matrices. weights[i] is the matrix
     * connecting layer i and layer i+1.
     */
    std::vector<Matrix> weights;

    /**
     * @brief A list of Bias matrices. biases[i] is the bias for layer i+1.
     */
    std::vector<Matrix> biases;

    /**
     * @brief A list of activation matrices. activations[i] stores the
     * output of layer i. This is needed for backpropagation.
     */
    std::vector<Matrix> activations;

    /**
     * @brief The learning rate for backpropagation.
     */

    std::vector<Matrix> weight_velocities;
    std::vector<Matrix> bias_velocities;   
    double training_rate;
    double momentum;

public:
    // --- Constructor ---

    /**
     * @brief Constructs a new Neural Network object.
     * @param topology A vector of integers defining the nodes in each layer,
     * from input to output (e.g., {4, 8, 16}).
     * @param learning_rate The learning rate to use for training.
     */
    NeuralNetwork(double learning_rate);

    void addLayer(int node_count, const std::string& activation);

    // --- Core Functions ---

    /**
     * @brief Feeds an input vector forward through the network.
     * @param input A Matrix (column vector, e.g., 4x1) of input data.
     * @return A Matrix (column vector, e.g., 16x1) of the network's output.
     */
    Matrix feedForward(const Matrix& input);

    /**
     * @brief Updates the network's weights and biases using backpropagation.
     * @param target The expected "correct" output for the last input.
     * @return The calculated loss (error) for this training step.
     */
    double update(const Matrix& target);

    // --- Utility Functions ---
    
    const Matrix& getActivationAt(int layer) const;

    /**
     * @brief Prints the dimensions of all weights and biases.
     * Useful for debugging.
     */
    void print() const;

};

#endif // NEURALNETWORK_H