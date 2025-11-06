#include <random>
#include <stdexcept>
#include <iomanip> 
#include <cmath> 
#include <cassert> 
#include "matrix.hpp"

Matrix::Matrix() : row(0), col(0) {
    // Default constructor: creates an empty 0x0 matrix.
    // The data vector is left empty by default.
}

Matrix::Matrix(int rows, int cols) : row(rows), col(cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive.");
    }
    // Resize the single vector to hold all elements, initialized to 0.0
    data.resize(rows * cols, 0.0);
}

int Matrix::getRows() const {
    return row;
}

int Matrix::getCols() const {
    return col;
}

double& Matrix::operator()(int r, int c) {
    if (r < 0 || r >= row || c < 0 || c >= col) {
        throw std::out_of_range("Matrix subscript out of bounds.");
    }
    return data[r * col + c];
}

const double& Matrix::operator()(int r, int c) const {
    if (r < 0 || r >= row || c < 0 || c >= col) {
        throw std::out_of_range("Matrix subscript out of bounds.");
    }
    return data[r * col + c];
}


// --- Utility Functions ---

void Matrix::print() const {
    std::cout << "Matrix (" << row << "x" << col << ")" << std::endl;
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            // Use the const operator() accessor
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << (*this)(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Matrix::randomize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    // Distribution between -1.0 and 1.0
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            // Use the operator() accessor to set the value
            (*this)(i, j) = dis(gen);
        }
    }
}

void Matrix::scale(double scalar) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            (*this)(i, j) *= scalar;
        }
    }
}

// --- Activation Functions ---

void Matrix::sigmoid() {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            double& val = (*this)(i, j); // Get reference
            val = 1.0 / (1.0 + exp(-val));
        }
    }
}

Matrix Matrix::dSigmoid() {
    Matrix result(row, col);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            // Derivative is: sigmoid(x) * (1 - sigmoid(x))
            // We assume 'this' matrix already has sigmoid applied.
            double val = (*this)(i, j);
            result(i, j) = val * (1.0 - val);
        }
    }
    return result;
}


// --- Static Matrix Operations ---

Matrix Matrix::add(const Matrix& a, const Matrix& b) {
    if (a.row != b.row || a.col != b.col) {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    Matrix result(a.row, a.col);
    for (int i = 0; i < a.row; ++i) {
        for (int j = 0; j < a.col; ++j) {
            // Use const accessor for a & b, non-const for result
            result(i, j) = a(i, j) + b(i, j);
        }
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& a, const Matrix& b) {
    if (a.row != b.row || a.col != b.col) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }
    Matrix result(a.row, a.col);
    for (int i = 0; i < a.row; ++i) {
        for (int j = 0; j < a.col; ++j) {
            result(i, j) = a(i, j) - b(i, j);
        }
    }
    return result;
}

Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    if (a.col != b.row) {
        throw std::invalid_argument("Matrix inner dimensions must match for multiplication.");
    }
    Matrix result(a.row, b.col);
    for (int i = 0; i < result.row; ++i) {
        for (int j = 0; j < result.col; ++j) {
            double sum = 0.0;
            for (int k = 0; k < a.col; ++k) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix Matrix::multiplyElementWise(const Matrix& a, const Matrix& b) {
    if (a.row != b.row || a.col != b.col) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication.");
    }
    Matrix result(a.row, a.col);
    for (int i = 0; i < a.row; ++i) {
        for (int j = 0; j < a.col; ++j) {
            result(i, j) = a(i, j) * b(i, j);
        }
    }
    return result;
}

Matrix Matrix::transpose(const Matrix& a) {
    Matrix result(a.col, a.row);
    for (int i = 0; i < a.row; ++i) {
        for (int j = 0; j < a.col; ++j) {
            result(j, i) = a(i, j);
        }
    }
    return result;
}

Matrix Matrix::fromVector(const std::vector<double>& vec) {
    Matrix result(vec.size(), 1);
    for (int i = 0; i < vec.size(); ++i) {
        result(i, 0) = vec[i];
    }
    return result;
}

std::vector<double> Matrix::toVector() const {
    if (col != 1) {
        std::cerr << "Warning: toVector() called on matrix with more than one column." << std::endl;
    }
    std::vector<double> result;
    for (int i = 0; i < row; ++i) {
        result.push_back((*this)(i, 0)); // Use const accessor
    }
    return result;
}


// --- Operator Overload Implementations ---

Matrix operator+(const Matrix& a, const Matrix& b) {
    return Matrix::add(a, b);
}

Matrix operator-(const Matrix& a, const Matrix& b) {
    return Matrix::subtract(a, b);
}

Matrix operator*(const Matrix& a, const Matrix& b) {
    return Matrix::multiply(a, b);
}