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
constexpr double n=1000.0;    //the number of data is n   
constexpr int int_n=n;
constexpr int int_m=m;
constexpr int mint=0.0;      //the minimum of theta[i]
using Data = Eigen::Matrix<double,int_m,int_n>;
using Mat = Eigen::Matrix<double, int_m, int_m>;
using Vec = Eigen::Matrix<double, int_m, 1>;

Data normal_dist(){
    Data X;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());

    // 平均0.0、標準偏差1.0で分布させる
    std::normal_distribution<> dist(0.0, 1.0);

    for (int k=0;k<n;k++) {
    // 正規分布で乱数を生成する
        for(int l=0;l<m;l++){
            X(l,k)=dist(engine);
        }
    }
    for(int k=0;k<n;k++){
        Vec V=X.col(k);
        for(int l=0;l<m;l++){
            X(l,k)=X(l,k)/V.norm();
        }
    }

    return X;
}

double kq(){
    double ans;
    ans=std::exp(-0.5*mint);
    return ans;
}

double p(Vec x,Vec theta,Vec gama){
    double ans=0.0;
    double expans;
    for(int k=0;k<m;k++){
        ans=ans-0.5*theta(k)*x(k)*x(k)+gama(k)*x(k);
    }
    expans=std::exp(ans);
    return expans;
}

int adapt(double prob){
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());

    // 一様実数分布
    // [-1.0, 1.0)の値の範囲で、等確率に実数を生成する
    std::uniform_real_distribution<> dist1(0.0, 1.0);

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
    Data X=normal_dist();
    Data Y=MatrixXd::Zero(m,n);
    int i=0;
    std::ofstream file("data.tsv");
    for(int k=0;k<n;k++){
        double prob=p(X.col(k),theta,gama)/kq();
        double choose=adapt(prob);
        if(choose==1){
            Y.col(i)=X.col(k);
            i=i+1;
        }
    }
    //std::cout << "The matrix x is:\n" << X << "\n\n";
    //std::cout << "The matrix Y is:\n" << Y << "\n\n";
    file << Y.transpose() << "\n";

    return 0;   
}

int main(){
    Vec theta;
    Vec gama;

    theta<<5.5,
           5.5,
           5.5;
    gama<<0.0,
          0.0,
          1.0;
    sampling(theta,gama);
    return 0;
}
