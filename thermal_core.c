/* Crank-Nicolson integration of a linear, time-dependent wet PV/T network.
   Units: capacities J/K, conductances W/K, temperatures Celsius, time seconds.
   Internal solid-fluid coupling persists at zero pump flow. */
#include <math.h>
static void coefficients(const double *u,const double *p,const double *e,double *a,double *b){
 double G=u[0],Ta=u[1],Tin=u[2],F=u[3],Tr=u[4],Ts=u[5],v=u[6];
 double H=p[3]+p[9]*v;
 a[0]=-(H+p[4]+p[7]+e[1]*G)/p[0];a[1]=p[4]/p[0];
 a[2]=p[4]/p[1];a[3]=-(p[4]+p[5]+p[6])/p[1];a[4]=p[6]/p[1];
 a[5]=p[6]/p[2];a[6]=-(p[6]+F)/p[2];
 b[0]=(G*(p[8]-e[0]+25*e[1]-e[2]*(G-1000)/1000)+H*Ta+p[7]*Ts)/p[0];
 b[1]=p[5]*Tr/p[1];b[2]=F*Tin/p[2];
}
void simulate(int n,const double *u,const double *p,const double *e,const double *y0,double dt,int sub,double *out){
 double y[3]={y0[0],y0[1],y0[2]},l[7],r[7],a[7],b[7],f[3],g[3];
 for(int j=0;j<3;j++)out[j]=y[j];
 double h=dt/sub/2;
 for(int i=0;i<n-1;i++){
  for(int k=0;k<sub;k++){
   for(int j=0;j<7;j++){
    double delta=u[(i+1)*7+j]-u[i*7+j];
    l[j]=u[i*7+j]+delta*k/sub;r[j]=u[i*7+j]+delta*(k+1)/sub;
   }
   coefficients(l,p,e,a,f);coefficients(r,p,e,b,g);
   double z0=y[0]+h*(a[0]*y[0]+a[1]*y[1]+f[0]+g[0]);
   double z1=y[1]+h*(a[2]*y[0]+a[3]*y[1]+a[4]*y[2]+f[1]+g[1]);
   double z2=y[2]+h*(a[5]*y[1]+a[6]*y[2]+f[2]+g[2]);
   /* Thomas elimination for (I-h*A_right) y_next = RHS. */
   double d0=1-h*b[0],c0=-h*b[1],l1=-h*b[2];
   double d1=1-h*b[3]-l1*c0/d0,c1=-h*b[4];
   z1-=l1*z0/d0;
   double l2=-h*b[5],d2=1-h*b[6]-l2*c1/d1;
   z2-=l2*z1/d1;
   y[2]=z2/d2;y[1]=(z1-c1*y[2])/d1;y[0]=(z0-c0*y[1])/d0;
  }
  for(int j=0;j<3;j++)out[(i+1)*3+j]=y[j];
 }
}
