clear;
clc;
close all;
%%   Grid coordinates                
N=50; 
M=1250;
dx=2/N; 
dt=1/M;     
x=-1:dx:1;  
t=0:dt:1;
r=dt/(dx^2); 
%%   Initial condition,boundary condition
N=length(x)-1;
M=length(t)-1;
Phi=ones(M+1,N+1);
Phi(1,:)=0;                
Phi(2:M+1,1)=0;             
Phi(2:M+1,N+1)=0;          
s=1-abs(x);
%%  Difference equation
for j=1:M
    for i=2:N
        Phi(j+1,i)=Phi(j,i)+r*(Phi(j,i+1)+Phi(j,i-1)-2*Phi(j,i))+dt*s(i);
    end
end
%%   Mapping
figure  
[x,t]=meshgrid(x,t);
mesh(x,t,Phi);
xlabel('x');
ylabel('t');
zlabel('\Phi(x,t)');
title('\Phi(x,t)');
view(75,50);  

load("Schrodinger.mat");
tt = grid(:,:,1);
xx = grid(:,:,2);
figure
mesh(xx,tt,pred);
view(75,50);  
title('\Phi(x,t) by INR');
