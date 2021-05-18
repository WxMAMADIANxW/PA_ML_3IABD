#include "library.h"
#include <stdio.h>
#include <iostream>
#include <Eigen>
#include <vector>
#include <random>
#include <experimental/random>


using Eigen::MatrixXf , Eigen::MatrixXd;
Eigen::MatrixXf create_linear_model(int input_dim){
    Eigen::MatrixXf model;

    return model;

}
int main()
{
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
    std::vector<float> test_vector = { 2,1,3 };
    Eigen::MatrixXf test = Eigen::Map<Eigen::Matrix<float, 3, 1> >(test_vector.data());
    std::cout << test  << std::endl;


}
