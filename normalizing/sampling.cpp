#include <random>
#include <fstream>
#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#include<complex>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>

using namespace Eigen;

constexpr double m=3.0;    //the dimension of parameter(theta gamma each dim=m)
constexpr double n=2000.0;    //the number of data is n   
constexpr int int_n=n;
constexpr int int_m=m;
using Mat = Eigen::Matrix<double, int_m, int_m>;
using Vec = Eigen::Matrix<double, int_m, 1>;

Vec normal_dist(){
    Vec x;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());

    // 平均0.0、標準偏差1.0で分布させる
    std::normal_distribution<> dist(0.0, 1.0);
    
    for(int l=0;l<m;l++){
        x(l)=dist(engine);
    }
    
    x=x/(x.norm());

    return x;
}

double p(Vec x,Vec theta,Vec gama){
    double ans=0.0;
    double expans;
    for(int k=0;k<m;k++){
        ans=ans-theta(k)*x(k)*x(k)+gama(k)*x(k);
    }
    expans=std::exp(ans);
    return expans;
}

double kq(Vec theta,Vec gama){
    double ans=0.0;
    for(int k=0;k<m;k++){
        ans=ans+gama(k)*gama(k)/(4*theta(k));
        //ans=ans+std::abs(gama(k));
    }
    ans=exp(ans)/2.5;
    return ans;
}

int adapt(double prob){
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());

    // 一様実数分布
    // [-1.0, 1.0)の値の範囲で、等確率に実数を生成する
    std::uniform_real_distribution<> dist1(0.0,1.0);

    for (size_t i = 0; i < n; ++i) {
        // 各分布法に基いて乱数を生成
        double r1 = dist1(engine);
        if(r1<=prob){
            return 1;
        }
        else{
            return 0;
        }
    }
}

int sampling(Vec theta,Vec gama){
    MatrixXd Y=MatrixXd::Zero(int_m,int_n);
    std::ofstream file("data.csv");
    int k=0;
    while(k<n){
        Vec x=normal_dist();
        //std::cout << "The vector x is:\n" << x << "\n\n";
        double prob=p(x,theta,gama)/kq(theta,gama);
        //printf("prop=%f\n",prob);
        double choose=adapt(prob);
        if(choose==1){
            Y.col(k)=x;
            //Y.col(k)(0)=x(0);
            //Y.col(k)(1)=x(1);
            //Y.col(k)(2)=-x(2);
            k=k+1;
        }
    }
    //std::cout << "The matrix x is:\n" << X << "\n\n";
    //std::cout << "The matrix Y is:\n" << Y << "\n\n";
    //file << Y << "\n";
    file << Y.transpose() << "\n";
    return 0;   
}

int main(){
    Vec theta;
    Vec gama;

    theta<<   
10,4,10;


    gama<<          
0,-1,0;

    sampling(theta,gama);
    return 0;
}
