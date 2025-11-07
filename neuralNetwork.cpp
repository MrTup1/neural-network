#include "neuralNetwork.hpp"
#include <stdexcept>
#include <iostream>

// --- Constructor ---

NeuralNetwork::NeuralNetwork(double learning_rate) {
    /*if (topology.size() < 2) {
        throw std::invalid_argument("Network must have at least an input and output layer.");
    }
    
    this->layer_nodes = topology;*/
    this->training_rate = learning_rate;

    /*// We need one fewer weight/bias matrix than we have layers.
    int num_weight_matrices = topology.size() - 1;
    
    weights.resize(num_weight_matrices);
    biases.resize(num_weight_matrices);
    
    // activations[0] will be the input, so resize for all layers
    activations.resize(topology.size());

    // Initialize the weights and biases
    for (int i = 0; i < num_weight_matrices; ++i) {
        // Get dimensions for this weight matrix
        int rows = topology[i+1]; // e.g., 8 hidden nodes
        int cols = topology[i];   // e.g., 4 input nodes
        
        // --- Initialize weights[i] ---
        // A (rows x cols) matrix, e.g., (8x4)
        weights[i] = Matrix(rows, cols);
        weights[i].randomize(); // Fill with random values [-1.0, 1.0]

        // --- Initialize biases[i] ---
        // A (rows x 1) column vector
        biases[i] = Matrix(rows, 1);
        biases[i].randomize(); // Fill with random values
    }*/
}

void NeuralNetwork::addLayer(int node_count, const std::string& activation) {
    // 1. Store the new layer's info
    layer_nodes.push_back(node_count);
    layer_activations.push_back(activation);

    // 2. Resize the 'activations' vector to make room for this layer's output
    //    We do this every time to keep it in sync with layer_nodes.
    activations.resize(layer_nodes.size());

    // 3. If this is the *first* layer (Input Layer), we don't create weights.
    //    We only create weights *connecting* layers.
    if (layer_nodes.size() > 1) {
        // This is a hidden or output layer.
        // We must create the weights and biases connecting the *previous* layer
        // to *this* new layer.
        
        int prev_layer_node_count = layer_nodes[layer_nodes.size() - 2];
        int curr_layer_node_count = node_count; // same as layer_nodes.back()

        // Create new weight matrix: (current_layer_nodes x prev_layer_nodes)
        Matrix w(curr_layer_node_count, prev_layer_node_count);
        w.randomize();
        weights.push_back(w); // Add to our list of weight matrices

        // Create new bias vector: (current_layer_nodes x 1)
        Matrix b(curr_layer_node_count, 1);
        b.randomize();
        biases.push_back(b); // Add to our list of bias matrices
    }
}

// --- Core Functions ---

Matrix NeuralNetwork::feedForward(const Matrix& input) {
    // Check if input dimensions are correct
    if (input.getRows() != layer_nodes[0] || input.getCols() != 1) {
        throw std::invalid_argument("Input matrix has incorrect dimensions for this network.");
    }

    // The first "activation" is the input itself
    activations[0] = input;

    // Loop through each layer (starting after the input layer)
    for (int i = 0; i < weights.size(); ++i) {        
        Matrix layer_output = weights[i] * activations[i]; 
        layer_output = layer_output + biases[i];
        std::string act_func = layer_activations[i + 1]; // +1 because [0] is input
        
        if (act_func == "sigmoid") {
            layer_output.sigmoid(); // Use in-place sigmoid
        }
        // --- BONUS (This is where you'd add more) ---
        // else if (act_func == "relu") {
        //     layer_output.relu(); 
        // }
        
        activations[i+1] = layer_output;
    }

    // Return reference to the final output (last activation)
    return activations.back();
}

double NeuralNetwork::update(const Matrix& target) {
    
    Matrix output = activations.back();
    Matrix error = target - activations.back();
    Matrix negativeError = activations.back() - target;
    Matrix squared_error = Matrix::multiplyElementWise(error, error);
    double total_loss = 0.5 * squared_error.sum();

    for (int i = weights.size() - 1; i >= 0; --i) {

        Matrix current_output = activations[i + 1];
        Matrix derivative;
        std::string act_func = layer_activations[i + 1]; // +1 because [0] is input

        if (act_func == "sigmoid") {
            derivative = Matrix::dsigmoid_nonDestructive(current_output);
        }
        // --- BONUS (This is where you'd add more) ---
        // else if (act_func == "relu") {
        //     derivative = Matrix::drelu_nonDestructive(current_output);
        // }
        else {
            // Default to sigmoid if unknown, or throw error
            derivative = Matrix::dsigmoid_nonDestructive(current_output);
        }
        // --- END REFACTORED LOGIC ---

        Matrix unscaled_gradient = Matrix::multiplyElementWise(derivative, negativeError);
        Matrix scaled_gradient = unscaled_gradient;
        scaled_gradient.scale(this->training_rate);

        Matrix prev_activation_T = Matrix::transpose(activations[i]);
        Matrix delta_weights = scaled_gradient * prev_activation_T;

        Matrix weights_T = Matrix::transpose(weights[i]);
        negativeError = weights_T * unscaled_gradient;
        
        
        //Remember to nudge gradient in opposite direction of gradient descent:
        weights[i] = weights[i] - delta_weights;
        biases[i] = biases[i] - scaled_gradient;
    }

    return total_loss; 
}

// --- Utility Functions ---
const Matrix& NeuralNetwork::getActivationAt(int layer) const {

    return activations[layer];
};

void NeuralNetwork::print() const {
    std::cout << "--- Network Topology ---" << std::endl;
    for (int i = 0; i < layer_nodes.size(); ++i) {
        std::cout << "Layer " << i << ": " << layer_nodes[i] << " nodes" << std::endl;
    }
    std::cout << "------------------------" << std::endl;

    for (int i = 0; i < weights.size(); ++i) {
        std::cout << "Weights (Layer " << i << " to " << i+1 << "): " 
                  << weights[i].getRows() << "x" << weights[i].getCols() << std::endl;
        std::cout << "Biases (for Layer " << i+1 << "): "
                  << biases[i].getRows() << "x" << biases[i].getCols() << std::endl;
    }
}