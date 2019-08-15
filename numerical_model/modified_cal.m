clear all;
close all;
load('C:\Users\nl4g1\OneDrive\Desktop\QMUL\BST data original\bstdata1400.mat')
load('C:\Users\nl4g1\OneDrive\Desktop\QMUL\MATLAB\tun_expdata.mat')
load('C:\Users\nl4g1\OneDrive\Desktop\QMUL\MATLAB\tun_expdata_mgo.mat')
load('C:\Users\nl4g1\OneDrive\Desktop\QMUL\MATLAB\tun_expdata_mgo2.mat')

x=[0.55];T_F=175;T=290;
Ta=42;
k1=50;
xi_S_pure=0.37;
xi_S_mg=[.7];
xi_S=xi_S_pure+xi_S_mg;
E=[0:5:30];
E_exp=tun_expdata(:,1)/1e5;
tunability_exp=tun_expdata(:,2)';
eplison_0=8.85e-12;
w_m=2.6e13;
w=[2*pi*1*1e4]';
f=w/2/pi;
% prompt1 = 'What is the tunability value? ';
% prompt2 = 'What is the loss value? ';
tun_exp=0.166;


loss_exp=0.00162;
k_mg=1;
aa=-0.442;bb=0.49;cc=-3.2;dd=0.453;

for j=1:length(E)
for k=1:length(x)
for m=1:length(xi_S)

C(k)=(0.78+0.76*x(k)^2)*10^5;

K_mg=exp(-k_mg*xi_S_mg);

K_pure=aa*tanh(bb*log10(f)+cc)+dd;

E_N(k)=(19+340*x(k)-50*x(k)^2-65*x(k)^3);

T_C(k,m)=Ta+439*x(k)-96*x(k)^2-k1*(xi_S(m))^2;

w_00(k)=0.67*(1+6*x(k))*1e13;

A2(k)=0.6*(1+20*x(k))^(-1);

w2=2*pi*30e9;

A3=0.05; w3=2*pi*10e9;

A4=0.02*K_mg; w4=2*pi*10e6/K_mg;

A5=0.01*K_mg; w5=2*pi*10e3/K_mg;

eta(k,m)=(T_F/(T_C(k,m))*sqrt(1/16+(T/T_F)^2)-1);

epsilon_00(k,m)=C(k)/T_C(k,m);

xi(j,m)=sqrt((xi_S(m)*19)^2+E(j)^2)/E_N(k);

a(j,k,m)=xi(j,m)^2+eta(k,m)^3;

if a(j,k,m)>=0
y(j,k,m)=(((a(j,k,m))^(1/2)+xi(j,m))^(1/3)-nthroot(((a(j,k,m)^(1/2)-xi(j,m))),3));
else
y(j,k,m)=(1/4*a(j,k,m)*xi(j,m)^(1/3)+xi(j,m)^(2/3)-3*eta(k,m))^(1/2);
end
G0(j,k,m)=(y(j,k,m)^2+eta(k,m))^(-1);

G(j,k,m)=K_pure*K_mg*(y(j,k,m)^2+eta(k,m))^(-1);


loss1(j,k,m)=-pi/8*w_00(k)/w_m^2*(T/T_C(k,m))^2*G0(j,k,m)^(1/2)*w*i;
loss2(j,k,m)=A2(k)/(1+i*w/w2)*(y(j,k,m)^2/(1+E(j)/E_N(k)));
loss3(m)=A3/(1+i*w/w3)*xi_S(m)^2;
loss4=A4/(1+i*w/w4); 
loss5=A5/(1+i*w/w5);

loss(j,k,m)=loss1(j,k,m)+loss2(j,k,m)...
    +loss3(m)+loss4+loss5;
epsilon_f(j,k,m)=epsilon_00(k)*(G(j,k,m)^(-1)+loss(j,k,m))^(-1);
d_correction=0.5;

% loss_ohmic=(dc_conduct/w+0.01*w*tau*(real(epsilon_f(j,k,m))*eplison_0)/(1+(w*tau)^2))/(eplison_0);
tan_loss(j,k,m)=abs(imag(epsilon_f(j,k,m))-d_correction)/real(epsilon_f(j,k,m));
nr_f(j,k,m)=1-real(epsilon_f(j,k,m))/real(epsilon_f(1,k,m));
epsilon_real(j,k,m)=real(epsilon_f(j,k,m));
end
end
end



eplison_sim=epsilon_real(:,:);
loss_sim=tan_loss(:,:);
nr_f_sim=nr_f(:,:);

data=[nr_f_sim,loss_sim,eplison_sim]

for m=1:length(xi_S)
 nr_max=nr_f_sim(:,m);
 loss_max=loss_sim(:,m);
 K(m)=nr_max(length(E))/loss_max(length(E));
end
for m=1:length(xi_S)
 nr_mean(m)=mean(nr_f_sim(:,m));
 loss_mean(m)=mean(loss_sim(:,m));
 K_mean(m)=nr_mean(m)/loss_mean(m);
end
[K_max,m_max]=max(K);
[K_max_mean,m_max_mean]=max(K_mean);
xi_S(m_max)
% xi_S(m_max_mean)
% K_max_mean
tan_loss;
[w/2/pi]';
epsilon_real(:,:)';
tan_loss(:,:)';



plot(E,epsilon_real(:,:))
figure
plot(E,tan_loss(:,:))
figure
plot(E,imag(epsilon_f(:,:)))
figure
plot(E,nr_f(:,:),tun_mgo(:,1),tun_mgo(:,2),'bo');
figure
plot(E,nr_f(:,:),tun_mgo2(:,1),tun_mgo2(:,6),'bo')
G
nr_f_sim
error_exp=1-(abs((loss_sim(1,1)-loss_exp)/loss_exp)+abs((nr_f_sim(j,1)-tun_exp)/tun_exp))/2;
for j=1:length(E)  
error(k,m)=sqrt(abs(sum((tun_mgo2(j,6)-nr_f(j))^2)/sum((nr_f(j)'-mean(nr_f'))^2)));
            
        %error(j2,j1,i2,i1,m2,m1)=sqrt(sum((qexpxlpe-nh(exptime/Ts,j2,j1,i2,i1,m2,m1)).^2));
end
1-error


% f_data=ones(j*k*m,1)*f;
% E_data=repmat(E,1,k*m)';
% 
% x_data=repmat(repelem(x,j)',m,1);
% def_data=repelem(xi_S_pure,j*k)';
% Mg_data=ones(j*k*m,1)*xi_S_mg;
% 
% tun_data=reshape(nr_f_sim,[],1);
% loss_data=reshape(loss_sim,[],1);
% xlswrite([num2str(xi_S_mg),'.xlsx'],f_data,5*xi_S_mg+1,'A1');
% xlswrite([num2str(xi_S_mg),'.xlsx'],E_data,5*xi_S_mg+1,'B1');
% xlswrite([num2str(xi_S_mg),'.xlsx'],def_data,5*xi_S_mg+1,'c1');
% xlswrite([num2str(xi_S_mg),'.xlsx'],x_data,5*xi_S_mg+1,'d1');
% xlswrite([num2str(xi_S_mg),'.xlsx'],Mg_data,5*xi_S_mg+1,'e1');
% xlswrite([num2str(xi_S_mg),'.xlsx'],tun_data,5*xi_S_mg+1,'f1');
% xlswrite([num2str(xi_S_mg),'.xlsx'],loss_data,5*xi_S_mg+1,'g1');
% Frequency=log10(fit(:,1));
% Permittivity=fit(:,2);
