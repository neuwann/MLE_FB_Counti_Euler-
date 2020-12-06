#include<stdio.h>
#include<stdlib.h>
#include <chrono>
#include<cmath>
#include<complex>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std::literals;

double omegad=0.9;  //the interval of omega which need to be decided properly maybe
double omegau=2.2;  //same as above
double d=1.0;  //d which depends on t0 need to be computed by hand
int N=200;
double t0=-1.0;  //need to be decided properly  

std::complex<double> f(double m,int dim,std::complex<double> x,VectorXd theta,VectorXd gama){
  std::complex<double> ans = 1.0;
  
  for(int i=0;i<m;i++){
    ans=ans*std::exp(gama(i)*gama(i)/(4.0*(theta(i)-1.0i*x-t0)))*std::pow(theta(i)-t0-1.0i*x,-0.5);
  } 
  return ans;
}

std::complex<double> F(double m,int dim,double omega,double p,double q,double h,VectorXd theta,VectorXd gama){
  std::complex<double> ans;
  ans=0.0;  
  for(int n=(-N-1);n<(N+1);n++){
    ans=ans+0.5*erfc(fabs(n*h)/p-q)*f(m,dim,n*h,theta,gama)*std::exp(-1.0i*omega*double(n)*h);
  }
  ans=h*ans;
  return ans;
}

std::complex<double> cf(double m,int dim,double p,double q,double h,VectorXd theta,VectorXd gama){
  std::complex<double> ans;
  ans=pow(M_PI,(m/2.0-1))*exp(-t0)*F(m,dim,1.0,p,q,h,theta,gama);
  return ans;
}

int main(){
    int range=300;
    int N=100;
    
    double h;
    double p;
    double q;
    double ans;
    double time;

    VectorXd theta0;
    VectorXd gama0;
    MatrixXd O;

        
    h=sqrt(2*M_PI*d*(omegad+omegau)/(omegad*omegad*N));
    p=sqrt(N*h/omegad);
    q=sqrt(omegad*N*h/4);

    for(int i=1;i<range;i++){
        double m=i;
        theta0=Eigen::VectorXd::Random(i);

        gama0=Eigen::VectorXd::Random(i);
  
        O=Eigen::MatrixXd::Identity(i,i);

        using namespace std;
        chrono::system_clock::time_point start, end;

        start = chrono::system_clock::now();

        ans=std::real(cf(m,i,p,q,h,theta0,gama0));  //pay attention to dim/2.0

        end = chrono::system_clock::now();

        double time = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0);
        printf("%d %lf\n",i,time);
 

        /*clock_t start = clock();

        ans=std::real(cf(m,i,p,q,h,theta0,gama0));  //pay attention to dim/2.0

        clock_t end = clock();

        const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
        //printf("time %lf[ms] %d\n", time,i);
        printf("%d %lf\n",i,time);
        //printf("%f\n",ans);
        //printf("%d %f\n",N,log(creal(c)));
        //printf("%d %f\n",int_m,time);*/
    }

    return 0;
}