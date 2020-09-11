#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#include<complex>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
using namespace Eigen;
using namespace std::literals;

constexpr double m=4.0;    //the dimension of theta
constexpr int int_m=m;    //the dimension of theta
double gama[int_m]={0.0,0.0,0.0,0.0};
constexpr double n=4.0;   //the size of X is n*m
constexpr double m=m;   //the size of X is n*m
constexpr int int_n=4;
constexpr int int_m=int_m;
double X[int_n][int_m]={{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
double O[int_m][int_m]={{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
using Mat = Eigen::Matrix<double, int_m, int_m>;
using Vec = Eigen::Matrix<double, int_m, 1>;

double omegad=0.9;    //the interval of omega which need to be decided properly maybe
double omegau=2.2;    //same as above
double d=1.0;    //d which depends on t0 need to be computed by hand
int N=100;
double t0=-1.0;    //need to be decided properly   

std::complex<double> f(std::complex<double> x,Vec theta){
    std::complex<double> ans = 1.0;
    
    for(int i=0;i<m;i++){
        ans=ans*std::exp(gama(i,0)*gama(i,0)/(4.0*(-theta(i,0)-1.0i*x-t0)))*std::pow(-theta(i,0)-t0-1.0i*x,-0.5);
    } 
    return ans;
}

int main(){
    double N_f;
    
    double h=sqrt(2*M_PI*d*(omegad+omegau)/(omegad*omegad*N));
    double p=sqrt(N*h/omegad);
    double q=sqrt(omegad*N*h/4);
    
    Vec theta;

    for(int k=0;k<m;k++){
        theta(k,0)=1;
    }
    std::complex<double> C;

    N_f=2.0*d*(omegad+omegau)*omegau*omegau/(M_PI*omegad*omegad);
    printf("N=%d\n",N);
    printf("h=%f\n",h);

    printf("f=%f\n",std::real(f(1,theta)));
    return 0;
}