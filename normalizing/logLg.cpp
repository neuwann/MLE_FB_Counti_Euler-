#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#include<complex>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std::literals;

//the size of X is n*m
constexpr double m=2.0;    //the dimension of parameter(theta gamma each dim=m)
constexpr double n=10000.0;    //the number of data is n   
constexpr int int_n=n;
constexpr int int_m=m;
double err=0.0001;
constexpr double cut=50.0;
constexpr int int_cut=cut;
constexpr int cutcut=int_cut*int_cut;
using Mat = Eigen::Matrix<double, int_m, int_m>;
using Mat_c = Eigen::Matrix<std::complex<double>, int_m, int_m>;
using Vec = Eigen::Matrix<double, int_m, 1>;
using Vec_c=Eigen::Matrix<std::complex<double>,int_m,1>;
using Mesh =Eigen::Matrix<double,int_cut,int_cut>;
using Plot =Eigen::Matrix<long double,cutcut,3>;
double omegad=0.9;    //the interval of omega which need to be decided properly maybe
double omegau=2.2;    //same as above
double d=1.0;    //d which depends on t0 need to be computed by hand
int N=100;
double t0=-1.0;    //need to be decided properly   

std::complex<double> f(std::complex<double> x,Vec theta,Vec gama){
    std::complex<double> ans = 1.0;
    
    for(int i=0;i<m;i++){
        ans=ans*std::exp(gama(i)*gama(i)/(4.0*(theta(i)-1.0i*x-t0)))*std::pow(theta(i)-t0-1.0i*x,-0.5);
    } 
    return ans;
}

std::complex<double> Pif(std::complex<double> x,Vec theta,Vec gama,int i){
    std::complex<double> ans=1.0;
    for(int k=0;k<m;k++){
        if(k!=i){
            ans=ans*std::exp(gama(k)*gama(k)/(4.0*(theta(k)-1.0i*x-t0)))*std::pow(theta(k)-1.0i*x-t0,-0.5);
        }
    }
    return ans;
}

Vec_c ft(std::complex<double> x,Vec theta,Vec gama){
    Vec_c ans;
    for(int i=0;i<m;i++){
        ans(i)=std::exp(gama(i)*gama(i)/(4.0*(theta(i)-1.0i*x-t0)))/(2.0*std::pow(theta(i)-1.0i*x-t0,1.5))*(-gama(i)*gama(i)/(2.0*(theta(i)-1.0i*x-t0))-1.0);
        ans(i)=ans(i)*Pif(x,theta,gama,i);
    } 

    return ans;
}

Vec_c fg(std::complex<double> x,Vec theta,Vec gama){
    Vec_c ans;
    for(int i=0;i<m;i++){
        ans(i)=std::exp(gama(i)*gama(i)/(4.0*(theta(i)-1.0i*x-t0)))*gama(i)*std::pow(theta(i)-1.0i*x-t0,-1.5)*0.5;    
        ans(i)=ans(i)*Pif(x,theta,gama,i);
    }   
    return ans;
}


std::complex<double> F(double omega,double p,double q,double h,Vec theta,Vec gama){
    std::complex<double> ans;
    ans=0.0;   
    for(int n=(-N-1);n<(N+1);n++){
        ans=ans+0.5*erfc(fabs(n*h)/p-q)*f(n*h,theta,gama)*std::exp(-1.0i*omega*double(n)*h);
    }
    ans=h*ans;
    return ans;
}

Vec_c Ft(double omega,double p,double q,double h,Vec theta,Vec gama){
    Vec_c ans;
    ans=VectorXcd::Zero(int_m); 
    for(int n=(-N-1);n<(N+1);n++){
        ans=ans+0.5*erfc(fabs(n*h)/p-q)*ft(n*h,theta,gama)*std::exp(-1.0i*omega*double(n)*h);
    }
    ans=h*ans;
    return ans;
}

Vec_c Fg(double omega,double p,double q,double h,Vec theta,Vec gama){
    Vec_c ans;
    ans=VectorXcd::Zero(int_m);
    for(int n=(-N-1);n<(N+1);n++){
        ans=ans+0.5*erfc(fabs(n*h)/p-q)*fg(n*h,theta,gama)*std::exp(-1.0i*omega*double(n)*h);
    }
    ans=h*ans;
    return ans;
}

std::complex<double> c(double p,double q,double h,Vec theta,Vec gama){
    std::complex<double> ans;
    ans=pow(M_PI,(m/2.0-1))*exp(-t0)*F(1.0,p,q,h,theta,gama);
    return ans;
}

Vec_c ct(double p,double q,double h,Vec theta,Vec gama){
    Vec_c ans;
    ans=pow(M_PI,(m/2.0-1))*exp(-t0)*Ft(1.0,p,q,h,theta,gama);
    return ans;
}

Vec_c cg(double p,double q,double h,Vec theta,Vec gama){
    Vec_c ans;
    ans=pow(M_PI,(m/2.0-1))*exp(-t0)*Fg(1.0,p,q,h,theta,gama);
    return ans;
}

double A_com(int i,int j,MatrixXd X){
    double ans=0; 
    for(int k=0;k<n;k++){
        ans=ans+X(i,k)*X(j,k);
    }
    ans=ans/n;
    return ans;
}

double B_com(int i,MatrixXd X){
    double ans=0;
    for(int k=0;k<n;k++){
        ans=ans+X(i,k);
    }
    ans=ans/n;
    return ans;
}

long double logL(double p,double q,double h,Vec theta,Vec gama,Mat O,Mat A,Vec B){
    //want to minimize this function
    std::complex<long double> ans;
    ans=std::log(c(p,q,h,theta,gama))+(A * (O.transpose())*(theta.asDiagonal())*O+O*B*(gama.transpose())).trace();  
    return std::real(ans);
}

Vec logLt(double p,double q,double h,Vec theta,Vec gama,Mat O,Mat A,Vec B){
    Vec ans;
    Mat OAOT=O*A*(O.transpose());
    std::complex<double> c_t;

    for(int k=0;k<m;k++){
        c_t=ct(p,q,h,theta,gama)(k);
        ans(k)=std::real(c_t/c(p,q,h,theta,gama))+OAOT(k,k);
    }
    return ans;
}

Vec logLg(double p,double q,double h,Vec theta,Vec gama,Mat O,Mat A,Vec B){
    Vec ans;
    Vec OB=O*B;

    for(int k=0;k<m;k++){
        ans(k)=std::real(cg(p,q,h,theta,gama)(k)/c(p,q,h,theta,gama)+OB(k));
    }
    return ans;
}

int plot(double p,double q,double h,Vec theta,Vec gama,Mat O,Mat A,Vec B,double bound){
    Mesh M;
    Vec G;
    Plot P;
    double interval=bound/cut;
    std::ofstream file("logLg.csv");
    for(int k=0;k<int_cut;k++){
        for(int l=0;l<int_cut;l++){
            G(0)=gama(0)+interval*(k-int_cut*0.5);
            G(1)=gama(1)+interval*(l-int_cut*0.5);
            Vec Lg=logLg(p,q,h,theta,G,O,A,B);
            M(k,l)=Lg(0);
            P(k*int_cut+l,0)=G(0);
            P(k*int_cut+l,1)=G(1);
            P(k*int_cut+l,2)=M(k,l);
        }
    }
    file << P << "\n";
    
    return 0;
}

Eigen::MatrixXd readCSV(std::string file, int rows, int cols) {

  std::ifstream in(file);
  
  std::string line;

  int row = 0;
  int col = 0;

  Eigen::MatrixXd res = Eigen::MatrixXd(rows, cols);

  if (in.is_open()) {

    while (std::getline(in, line)) {

      char *ptr = (char *) line.c_str();
      int len = line.length();

      col = 0;

      char *start = ptr;
      for (int i = 0; i < len; i++) {

        if (ptr[i] == ' ') {
          res(row, col++) = atof(start);
          start = ptr + i + 1;
        }
      }
      res(row, col) = atof(start);

      row++;
    }

    in.close();
  }
  return res;
}



int main(){
    double N_f;
    
    double h=sqrt(2*M_PI*d*(omegad+omegau)/(omegad*omegad*N));
    double p=sqrt(N*h/omegad);
    double q=sqrt(omegad*N*h/4);
    
        
    MatrixXd X;
    X=readCSV("data.csv",int_m,int_n);

    Mat A;
    for(int k=0;k<m;k++){
        for(int l=0;l<m;l++){
            A(k,l)=A_com(k,l,X);
        }
    }
    std::cout << "The matrix A is:\n" << A << "\n\n";
    Vec B;
    for(int k=0;k<m;k++){
        B(k,0)=B_com(k,X);
    }
    std::cout << "The vector B is:\n" << B << "\n\n";

    Vec theta0;
    Vec gama0;
    Mat O;

    theta0 << 2.0,
              2.0;

    gama0 << 2.0,
             0.0;
    
    Vec theta_ex;
    theta_ex << 4.0,
                4.0;
    
    Vec gama_ex;
    gama_ex << 2.0,
               0.0;

    O=MatrixXd::Identity(int_m,int_m);

    std::complex<double> C;

    N_f=2.0*d*(omegad+omegau)*omegau*omegau/(M_PI*omegad*omegad);
    plot(p,q,h,theta0,gama0,O,A,B,10.0);
    printf("c=%f\n",std::real(c(p,q,h,theta_ex,gama_ex)));
    printf("ct=%f\n",std::real(ct(p,q,h,theta_ex,gama_ex)(0)));
    printf("logLt=%f\n",logLt(p,q,h,theta_ex,gama_ex,O,A,B)(0));
    return 0;
}