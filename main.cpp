#include <vector>
#include <iostream>
#include <random>
#include "matrix.hpp"

int main() {
    std::cout << "--- Matrix Class Test Program ---" << std::endl << std::endl;

    //Test Manual setup:
    std::cout << "Manual setting matrix values of 'C':" <<std::endl;
    Matrix c(2, 2);
    c(0, 0) = 1.0;
    c(0, 1) = 2.0;
    c(1, 0) = 3.0;
    c(1, 1) = 4.0;
    c.print();

    // --- 1. Test Constructor and Randomize ---
    std::cout << "1. Creating and randomizing a 3x3 matrix 'A'..." << std::endl;
    Matrix a(3, 3);
    a.randomize();
    a.print();

    std::cout << "2. Creating and randomizing a 3x3 matrix 'B'..." << std::endl;
    Matrix b(3, 3);
    b.randomize();
    b.print();

    // --- 2. Test Matrix Addition ---
    std::cout << "3. Testing Matrix Addition (A + B)..." << std::endl;
    try {
        Matrix c = a + b; // Uses operator+
        c.print();
    } catch (const std::exception& e) {
        std::cerr << "Addition Error: " << e.what() << std::endl;
    }

    // --- 3. Test Matrix Multiplication ---
    std::cout << "4. Testing Matrix Multiplication (A * B)..." << std::endl;
    try {
        Matrix d = a * b; // Uses operator*
        d.print();
    } catch (const std::exception& e) {
        std::cerr << "Multiplication Error: " << e.what() << std::endl;
    }

    // --- 4. Test Transpose ---
    std::cout << "5. Testing Transpose (transpose(A))..." << std::endl;
    try {
        Matrix a_t = Matrix::transpose(a);
        a_t.print();
    } catch (const std::exception& e) {
        std::cerr << "Transpose Error: " << e.what() << std::endl;
    }
    
    // --- 5. Test Non-matching dimensions ---
    std::cout << "6. Testing error handling (3x3 + 2x2)..." << std::endl;
    try {
        Matrix e(2, 2);
        Matrix f = a + e;
        f.print(); // This shouldn't be reached
    } catch (const std::exception& e) {
        std::cerr << "Caught Expected Error: " << e.what() << std::endl << std::endl;
    }
    /*
    // --- 6. Test Sigmoid ---
    std::cout << "7. Testing Sigmoid function on matrix 'A'..." << std::endl;
    a.sigmoid();
    a.print();
    
    std::cout << "8. Testing dSigmoid function on (sigmoided) matrix 'A'..." << std::endl;
    Matrix a_dsig = a.dSigmoid();
    a_dsig.print();
    */
    std::cout << "9. Testing scaling on matrix 'A'..." << std::endl;  
    Matrix a_Scale = a;
    a_Scale.scale(3.0);
    a_Scale.print();

    std::cout << "10. Element wise multplication on matrix 'A'and 'B'..." << std::endl;  
    Matrix a_ElementWise = a.multiplyElementWise(a, b);
    a_ElementWise.print();

    std::cout << "--- Test Complete ---" << std::endl;

    return 0;
}