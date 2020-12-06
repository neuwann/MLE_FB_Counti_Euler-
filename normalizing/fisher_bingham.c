#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<complex.h>

int dim=3.0;    //the dimension of theta(parameter)
double t0=-1.0;    //need to be decided properly    
double theta[3]={100.0,100.0,100.0};
double gama[3]={0.0,0.0,0.0};       
double omegad=0.9;    //the interval of omega which need to be decided properly maybe
double omegau=2.2;    //same as above
double d=1.0;    //d which depends on t0 need to be computed by hand

double _Complex f(double _Complex x){
    double _Complex ans;
    ans=1.0+0*I;
    
    for(int i=1;i<(dim+1);i++){
        ans=ans*cexp(gama[i-1]*gama[i-1]/(4*(theta[i-1]-I*x-t0)))*cpow(theta[i-1]-t0-I*x,-0.5);
    } 
    return ans;
}

double _Complex F(double omega,double p,double q,double h,int N){
    double _Complex ans;
    ans=0.0;
    
    for(int n=(-N-1);n<(N+1);n++){
        ans=ans+0.5*erfc(fabs(n*h)/p-q)*f(n*h)*cexp(-I*omega*n*h);
    }
    ans=h*ans;
    return ans;
}

int main(){
    double N_f;
    double _Complex c;
    int N=200;
    
    double h;
    double p;
    double q;


    h=sqrt(2*M_PI*d*(omegad+omegau)/(omegad*omegad*N));
    p=sqrt(N*h/omegad);
    q=sqrt(omegad*N*h/4);
    c=pow(M_PI,(dim/2.0-1))*exp(-t0)*F(1.0,p,q,h,N);  //pay attention to dim/2.0
    printf("%d %f\n",N,creal(c));
    //printf("%f\n",creal(F(1,p,q,h,N)));
    
    return 0;
}