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
constexpr double m=6.0;  //the dimension of parameter(theta gamma each dim=m)
constexpr double n=5949.0;  //the number of data is n  
constexpr int int_n=n;
constexpr int int_m=m;
double err=0.00001;
using Mat = Eigen::Matrix<double, int_m, int_m>;
using Mat_c = Eigen::Matrix<std::complex<double>, int_m, int_m>;
using Vec = Eigen::Matrix<double, int_m, 1>;
using Vec_c=Eigen::Matrix<std::complex<double>,int_m,1>;

double omegad=0.9;  //the interval of omega which need to be decided properly maybe
double omegau=2.2;  //same as above
double d=1.0;  //d which depends on t0 need to be computed by hand
int N=100;
double t0=-1.0;  //need to be decided properly  

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

double logL(double p,double q,double h,Vec theta,Vec gama,Mat O,Mat A,Vec B){
  //want to minimize this function
  std::complex<double> ans;
  ans=std::log(c(p,q,h,theta,gama))+(A * (O.transpose())*(theta.asDiagonal())*O+O*B*(gama.transpose())).trace(); 
  return std::real(ans);
}

Vec logLt(double p,double q,double h,Vec theta,Vec gama,Mat O,Mat A,Vec B){
  Vec ans;
  Mat OAOT=O*A*(O.transpose());

  for(int k=0;k<m;k++){
    ans(k)=std::real(ct(p,q,h,theta,gama)(k)/c(p,q,h,theta,gama)+OAOT(k,k));
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

Mat f_A(Vec theta,Vec gama,Mat O,Mat A,Vec B){
  Mat ans;
  ans=(theta.asDiagonal())*O*A*(O.transpose())-O*A*(O.transpose())*(theta.asDiagonal())+gama*(B.transpose())*(O.transpose());
  return ans;
}

Mat evt(double t,Vec theta,Vec gama,Mat O,Mat A,Vec B){
  Mat ans; 
  Mat v;

  v=f_A(theta,gama,O,A,B)-(f_A(theta,gama,O,A,B).transpose()); 
  ans=(v*t).exp();
  return ans;
}

int Alg(double err,double p,double q,double h,Vec theta,Vec gama,Mat O,Mat A,Vec B){
  double dt=0.1,errt=1;
  double dg=0.1,errg=1;
  double dO=0.1,errO=0;
  double lt,lg;
  int ctt=0;
  int ctg=0;
  int ctO=0;
  
  Vec pretheta;
  Vec pregama;
  Mat preO;
  Vec n_gama;
  
  pretheta=theta; 

  while(std::abs(errt)>err || std::abs(errg)>err || std::abs(errO)>err){
    dt=1.0;
    dg=1.0;
    dO=1.0;

    while(true){
      pretheta=theta-logLt(p,q,h,theta,gama,O,A,B)*dt;

      while(pretheta(0)<=0 || pretheta(1)<=0){
         dt=0.5*dt;
         pretheta=theta-logLt(p,q,h,theta,gama,O,A,B)*dt;
         ctt=ctt+1;
         if(ctt>20){
           ctt=0;
           printf("0break\n");
           return 0; 
         }
      }

      errt=logL(p,q,h,pretheta,gama,O,A,B)-logL(p,q,h,theta,gama,O,A,B);
      //double lt=logLt(p,q,h,theta,gama,O,A,B)(0);
      //std::cout << "logLt is:\n" << logLt(p,q,h,theta,gama,O,A,B) << "\n";
      //printf("errt=%f\n",errt);
      if(errt<=0){
        ctt=0;
        theta=pretheta;
        //printf("steptg=%f\n",dt);
        //std::cout << "theta is:\n" << theta << "\n";
        break;
      }
      else{
        dt=0.5*dt;
        ctt=ctt+1;
        if(ctt>20){
          ctt=0;
          //printf("break\n");
          break;
        }
        //printf("steptb=%f\n",dt);
      }
    }
    while(true){
      pregama=gama-logLg(p,q,h,theta,gama,O,A,B)*dg;

      errg=logL(p,q,h,theta,pregama,O,A,B)-logL(p,q,h,theta,gama,O,A,B);
      //printf("errg=%f\n",errg);
      if(errg<=0){
        ctg=0;
        gama=pregama;
        //printf("stepgg=%f\n",dg);
        //std::cout << "gama is:\n" << -gama << "\n";
        break;
      }
      else{
        dg=0.5*dg;
        ctg=ctg+1;
        if(ctg>20){
          ctg=0;
          //printf("break\n");
          break;
        }
        //printf("stepgb=%f\n",dg);
      }
    }
    /*while(true){
      preO=evt(dO,theta,gama,O,A,B)*O;
      errO=logL(p,q,h,theta,gama,preO,A,B)-logL(p,q,h,theta,gama,O,A,B);
      printf("errO=%f\n",errO);
      if(errO<=0){
        ctO=0;
        O=preO;
        printf("stepOg=%f\n",dO);
        break;
      }
      else{
        dO=0.5*dO;
        ctO=ctO+1;
        if(ctO>25){
          ctO=0;
          printf("break\n");
          break;
        }
        //printf("stepOb=%f\n",dO);
      }
    }*/
  }
  
  n_gama=gama/gama.norm();

  std::cout << "theta is:\n" << theta << "\n";
  std::cout << "gama is:\n" << -gama << "\n";
  //std::cout << "n_gama is:\n" << n_gama << "\n";
  //std::cout << "The matrix O is:\n" << O << "\n\n";

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
  
  MatrixXd XT;
  XT=readCSV("label_latent_vars9.csv",int_n,int_m);

  MatrixXd X;
  X=XT.transpose();
  //X=readCSV("data.csv",int_m,int_n);

  Mat A;
  for(int k=0;k<m;k++){
    for(int l=0;l<m;l++){
      A(k,l)=A_com(k,l,X);
    }
  }
  //std::cout << "The matrix A is:\n" << A << "\n\n";
  Vec B;
  for(int k=0;k<m;k++){
    B(k,0)=B_com(k,X);
  }
  //std::cout << "The vector B is:\n" << B << "\n\n";

  Vec theta0;
  Vec gama0;
  Mat O;

  theta0 << 10,10,10,10,10,10;

  gama0 << 1,1,1,1,1,1;
  
  O=Eigen::MatrixXd::Identity(int_m,int_m);


  N_f=2.0*d*(omegad+omegau)*omegau*omegau/(M_PI*omegad*omegad);
  Alg(err,p,q,h,theta0,gama0,O,A,B);
  return 0;
}