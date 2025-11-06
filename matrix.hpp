#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <random>
#include <stdexcept> // For std::out_of_range

class Matrix 
{
    private:
        std::vector<double> data;
        int row;
        int col;
    public:
        Matrix();
        Matrix(int rows, int cols);

        int getRows() const;
        int getCols() const;

        double& operator()(int r, int c); //To get data position, since not using vector of vectors
        const double& operator()(int r, int c) const;

        void print() const; //print function for debugging matrix content 
        void randomize(); //generate random values for the starting matrix

        void scale(double scalar); //Scales all values within the matrix

        void sigmoid();

        Matrix dSigmoid();

        static Matrix add(const Matrix& a, const Matrix& b);
        static Matrix subtract(const Matrix& a, const Matrix& b);
        static Matrix multiply(const Matrix& a, const Matrix& b);
        static Matrix multiplyElementWise(const Matrix& a, const Matrix& b);
        static Matrix transpose(const Matrix& a);

        static Matrix fromVector(const std::vector<double>& vec);
        std::vector<double> toVector() const;
    };
    
    
Matrix operator+(const Matrix& a, const Matrix& b);

Matrix operator-(const Matrix& a, const Matrix& b);

Matrix operator*(const Matrix& a, const Matrix& b);

#endif // MATRIX_H