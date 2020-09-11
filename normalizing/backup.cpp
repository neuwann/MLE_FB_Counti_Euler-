#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#include<complex>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
using namespace Eigen;
using namespace std::literals;

constexpr double dim=4.0;    //the dimension of theta
constexpr int int_dim=dim;    //the dimension of theta
double theta[int_dim]={0.0,-1.0,-2.0,-5.0};
double gama[int_dim]={0.0,0.0,0.0,0.0};
constexpr double n=4.0;   //the size of X is n*m
constexpr double m=dim;   //the size of X is n*m
constexpr int int_n=4;
constexpr int int_m=int_dim;
double X[int_n][int_m]={{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
double O[int_dim][int_dim]={{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
using Mat = Eigen::Matrix<double, int_dim, int_dim>;

double omegad=0.9;    //the interval of omega which need to be decided properly maybe
double omegau=2.2;    //same as above
double d=1.0;    //d which depends on t0 need to be computed by hand
int N=100;
double t0=-1.0;    //need to be decided properly   

std::complex<double> f(std::complex<double> x){
    std::complex<double> ans = 1.0;
    
    for(int i=0;i<dim;i++){
        ans=ans*std::exp(gama(i,0)*gama(i,0)/(4.0*(-theta(i,0)-1.0i*x-t0)))*std::pow(-theta(i,0)-t0-1.0i*x,-0.5);
    } 
    return ans;
}

std::complex<double> ft(std::complex<double> x,int i){
    std::complex<double> ans;

    ans=std::exp(gama(i,0)*gama(i,0)/(4.0*(-theta(i,0)-1.0i*x-t0)))/(2.0*std::pow(-theta(i,0)-1.0i*x-t0,1.5))*(gama(i,0)+gama(i,0)/(2.0*(-theta(i,0)-1.0i*x-t0)+1.0));
    
    return ans;
}

std::complex<double> fg(std::complex<double> x,int i){
    std::complex<double> ans;

    ans=std::exp(gama(i,0)*gama(i,0)/(4.0*(-theta(i,0)-1.0i*x-t0)))*2.0*gama(i,0)*std::pow(-theta(i,0)-1.0i*x-t0,-0.5);
    
    return ans;
}


std::complex<double> F(double omega,double p,double q,double h){
    std::complex<double> ans;
    ans=0.0;
    
    for(int n=(-N-1);n<(N+1);n++){
        ans=ans+0.5*erfc(fabs(n*h)/p-q)*f(n*h)*std::exp(-1.0i*omega*double(n)*h);
    }
    ans=h*ans;
    return ans;
}

std::complex<double> Ft(double omega,double p,double q,double h,int i){
    std::complex<double> ans;
    ans=0.0;
    
    for(int n=(-N-1);n<(N+1);n++){
        ans=ans+0.5*erfc(fabs(n*h)/p-q)*ft(n*h,i)*std::exp(-1.0i*omega*double(n)*h);
    }
    ans=h*ans;
    return ans;
}

std::complex<double> Fg(double omega,double p,double q,double h,int i){
    std::complex<double> ans;
    ans=0.0;
    
    for(int n=(-N-1);n<(N+1);n++){
        ans=ans+0.5*erfc(fabs(n*h)/p-q)*fg(n*h,i)*std::exp(-1.0i*omega*double(n)*h);
    }
    ans=h*ans;
    return ans;
}

std::complex<double> c(double p,double q,double h){
    std::complex<double> ans;
    ans=pow(M_PI,(dim/2.0-1))*exp(-t0)*F(1.0,p,q,h);
    return ans;
}

std::complex<double> ct(double p,double q,double h,int i){
    std::complex<double> ans;
    ans=pow(M_PI,(dim/2.0-1))*exp(-t0)*Ft(1.0,p,q,h,i);
    return ans;
}

std::complex<double> cg(double p,double q,double h,int i){
    std::complex<double> ans;
    ans=pow(M_PI,(dim/2.0-1))*exp(-t0)*Fg(1.0,p,q,h,i);
    return ans;
}

double A(int i,int j){
    double ans=0;
    
    for(int k=0;k<n;k++){
        ans=ans+X[i][k]*X[j][k];
    }
    ans=ans/n;
    return ans;
}

double B(int i){
    double ans=0;

    for(int k=0;k<n;k++){
        ans=ans+X[i][k];
    }
    ans=ans/n;
    return ans;
}

double BTOT(int i){
    double ans=0;

    for(int k=0;k<n;k++){
        ans=ans+B(k)*O[i][k];
    }
    return ans;
}

double OA(int i,int j){
    double ans=0;

    for(int k=0;k<m;k++){
        ans=ans+O[i][k]*A(k,j);
    }
    return ans;
}

double OAOT(int i,int j){
    double ans=0;

    for(int k=0;k<m;k++){
        ans=ans+OA(i,k)*O[j][k];
    }
    return ans;
}

double diag(int i){
    double ans;
    ans=OAOT(i,i);
    return ans;
}

std::complex<double> logLt(double p,double q,double h,int i){
    std::complex<double> ans;
    ans=ct(p,q,h,i)/c(p,q,h)+diag(i);  

    return ans;
}

std::complex<double> logLg(double p,double q,double h,int i){
    std::complex<double> ans;
    ans=cg(p,q,h,i)/c(p,q,h)+BTOT(i);  

    return ans;
}

double f_A(int i,int j){
    double ans;
    ans=theta(i,0)*OAOT(i,j)-OAOT(i,j)*theta[j]+gama(i,0)*BTOT(j);
    return ans;
}

double v(int i,int j){
    double ans;
    ans=f_A(i,j)-f_A(j,i);
    return ans;
}

double evt(int i,int j,double t){
    double ans;
    Mat A;

    for(int k=0;k<m;k++){
        for(int l=0;l<m;l++){
            A(k,l)=f_A(k,l)*t;
        }
    }
    Mat B=A.exp();
    ans=B(i,j);
    return ans;
}

int main(){
    double N_f;
    
    double h=sqrt(2*M_PI*d*(omegad+omegau)/(omegad*omegad*N));
    double p=sqrt(N*h/omegad);
    double q=sqrt(omegad*N*h/4);
    
    std::complex<double> C;

    N_f=2.0*d*(omegad+omegau)*omegau*omegau/(M_PI*omegad*omegad);
    printf("N=%d\n",N);
    printf("h=%f\n",h);

    C=c(p,q,h);
    printf("cr=%f\n",std::real(C));
    printf("ci=%f\n",std::imag(C));

    printf("logLt=%f\n",std::real(logLt(p,q,h,0)));
    printf("logLg=%f\n",std::real(logLg(p,q,h,0)));

    printf("evt=%f\n",evt(3,3,1));
    return 0;
}