#include <random>
#include <fstream>
#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#include<complex>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>

using namespace Eigen;

int main(){
    Eigen::MatrixXd A=Eigen::MatrixXd::Identity(100,100);
    Eigen::MatrixXd B=A;
    Eigen::MatrixXd C=A*B;
    printf("ans=%f\n",C(0,0));
    return 0;
}