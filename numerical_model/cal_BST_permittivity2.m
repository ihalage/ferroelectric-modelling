clear all;
close all;
load('C:\Users\nl4g1\OneDrive\Desktop\QMUL\BST data original\bstdata1500.mat')

x=0.6;T_F=175;T=[290];
Ta=[42];
xi_S_mg=0.0;
xi_S_pure=[0.4];
xi_S=xi_S_pure+xi_S_mg;
k1=50;
T_C=Ta+439*x-96*x^2-k1*xi_S.^2;
C=(0.78+0.76*x^2)*10^5;
E_N=19+340*x-50*x^2-65*x^3;
E=[0];
w_00=0.67*(1+6*x)*1e13; w_m=2.6e13;
% w=2*pi*[8:0.02:12]*1e9;
 w=2*pi*[[1:0.1:9]*1e3,[1:9]*1e4,[1:0.1:9]*1e5,[1:0.1:9]*1e6,[1:0.1:9]*1e7,[1:0.1:9]*1e8,[1:0.1:9]*1e9,[1:0.1:5]*1e10];
f=w/2/pi;
% 1e3,3e3,5e3,1e4,3e4,5e4,1e5,3e5,5e5,1e6,3e6,5e6,1e7,3e7,5e7,1e8,5e8,1e9,5e9,1e10;
k_mg=1.0;
aa=-0.442;bb=0.49;cc=-3.2;dd=0.453;
K_mg=exp(-k_mg*xi_S_mg);
E_N=(19+340*x-50*x^2-65*x^3);
A2=0.6*(1+20*x)^(-1);
w2=2*pi*30e9;
A3=0.05; w3=2*pi*10e9;
A4=0.02*K_mg; w4=2*pi*10e6/K_mg;
A5=0.01*K_mg; w5=2*pi*10e3/K_mg;

for j=1:length(T)
for k=1:length(w)
for m=1:length(xi_S)
eta(j,m)=T_F/(T_C(m))*sqrt(1/16+(T(j)/T_F)^2)-1;
epsilon_00(m)=C/T_C(m);

K_pure(k)=aa*tanh(bb*log10(f(k))+cc)+dd;
xi(m)=sqrt((xi_S(m)*19)^2+E^2)/E_N;

a(j,m)=xi(m)^2+eta(j,m)^3;


if a(j,m)>=0
y(j,m)=(((a(j,m))^(1/2)+xi(m))^(1/3)-nthroot(((a(j,m)^(1/2)-xi(m))),3));
else
y(j,m)=(1/4*a(j,m)*xi(m)^(1/3)+xi(m)^(2/3)-3*eta(j,m))^(1/2);
end
G0(j,k,m)=(y(j,m)^2+eta(j,m))^(-1);
G(j,k,m)=K_pure(k)*K_mg*(y(j,m)^2+eta(j,m))^(-1);

loss1(j,k,m)=-pi/8*w_00/w_m^2*(T(j)/T_C(m))^2*G0(j,k,m)^(1/2)*w(k)*i;
loss2(j,k,m)=A2/(1+i*w(k)/w2)*(y(j,m)^2/(1+E/E_N));
loss3(k,m)=A3/(1+i*w(k)/w3)*xi_S(m)^2;
loss4(k)=A4/(1-i*w(k)/w4); 
loss5(k)=A5/(1-i*w(k)/w5);

loss(j,k,m)=loss1(j,k,m)+loss2(j,k,m)...
    +loss3(k,m)+loss4(k)+loss5(k);
epsilon_f(j,k,m)=epsilon_00(m)*(G(j,k,m)^(-1)-loss(j,k,m))^(-1);
d_correction=0.5;
tan_loss(j,k,m)=abs(imag(epsilon_f(j,k,m))-d_correction)/real(epsilon_f(j,k,m));
nr_f(j,k,m)=1-real(epsilon_f(j,k,m))/real(epsilon_f(j,k,m));
epsilon_real(j,k,m)=real(epsilon_f(j,k,m));
end
end
end
exp_freq=bstdata(:,1);
for j=1:length(T)
[~,I(j)]=min(abs(exp_freq-T(j)));
exp_permittivity(j)=bstdata(I(j),2);
end

for k=1:length(w)
for m=1:length(xi_S)  
error(k,m)=abs(sum((exp_permittivity'-epsilon_real(:,k,m)).^2)/sum((exp_permittivity'-mean(exp_permittivity')).^2));
            
        %error(j2,j1,i2,i1,m2,m1)=sqrt(sum((qexpxlpe-nh(exptime/Ts,j2,j1,i2,i1,m2,m1)).^2));
end
end
eplison_sim=epsilon_real(:,:);
loss_sim=tan_loss(:,:);

[error_min, minind] = min(error(:));
[kfit,mfit] = ind2sub(size(error),minind);
1-error_min
tan_loss;
[w/2/pi]';
epsilon_real(:,:)';
tan_loss(:,:)';
plot(T,epsilon_real(:,:))
figure
plot(T,a(:,:));
figure
plot(T,G(:,:));
figure
plot(T,y(:,:));
figure
plot(T,epsilon_real(:,kfit,mfit),T,exp_permittivity,'blacko','LineWidth',2)
figure
plot(T,tan_loss(:,:))
figure
plot(T,imag(epsilon_f(:,:)))
figure
semilogx(f,epsilon_real(:,:))
figure
semilogx(f,tan_loss(:,:))
xi_S(mfit)
T_C(mfit)-273.15
data=[T',epsilon_real(:,kfit,mfit),exp_permittivity'];
% for m=1:length(xi_S)
% xlswrite('C:\Users\ning\Desktop\data2.xlsx',real(epsilon_f(:,1,m))',1,strcat('B',num2str(m+1),:,'zz',num2str(m+length(m+1))))
% end
% for m=1:length(xi_S)
% xlswrite('C:\Users\ning\Desktop\data2.xlsx',tan_loss(:,1,m)',2,strcat('B',num2str(m+1),:,'zz',num2str(m+length(m+1))))
% end
% for m=1:length(xi_S)
% xlswrite('C:\Users\ning\Desktop\data2.xlsx',imag(epsilon_f(:,1,m))',3,strcat('B',num2str(m+1),:,'zz',num2str(m+length(m+1))))
% end
% for k=1:length(Ta)
% xlswrite('C:\Users\ning\Desktop\data3.xlsx',real(epsilon_f(:,k,1))',1,strcat('B',num2str(k+1),:,'zz',num2str(k+1)))
% end
% for k=1:length(Ta)
% xlswrite('C:\Users\ning\Desktop\data3.xlsx',tan_loss(:,k,1)',2,strcat('B',num2str(k+1),:,'zz',num2str(k+1)))
% end
% for k=1:length(Ta)
% xlswrite('C:\Users\ning\Desktop\data3.xlsx',imag(epsilon_f(:,k,1))',3,strcat('B',num2str(k+1),:,'zz',num2str(k+1)))
% end
% xlswrite('C:\Users\ning\Desktop\data.xlsx',[w/2/pi]',1,'A2:A17')
% xlswrite('C:\Users\ning\Desktop\data.xlsx',E/10,1,'B1:L1')
% xlswrite('C:\Users\ning\Desktop\data.xlsx',epsilon_real(:,:)',1,'B2:L17')

% xlswrite('C:\Users\ning\Desktop\data.xlsx',[w/2/pi]',1,'A2:A17')
% xlswrite('C:\Users\ning\Desktop\data.xlsx',[T],1,'B1:H1')
% for m=1:length(w)
% xlswrite('C:\Users\ning\Desktop\data.xlsx',real(epsilon_f(1,:)),1,strcat('B',num2str(m+1),:,'Q',num2str(m+1)))
% end
% 
% for m=1:length(w)    
% plot(T,real(epsilon_f(1,:)))
% hold on
% end
% hold off
% figure
% 
% 
% for m=1:length(w) 
% for j=1:length(T)
% aaa(j,k,m)=real(epsilon_f(1,j));
% end
% end
% 
% for j=1:length(T)
% semilogx(w/2/pi,aaa(j,:))
% hold on;
% end
% hold off
% figure
% 
% xlswrite('C:\Users\ning\Desktop\data.xlsx',[w/2/pi]',6,'A2:A17')
% xlswrite('C:\Users\ning\Desktop\data.xlsx',[T],6,'B1:H1')
% for m=1:length(w)
% xlswrite('C:\Users\ning\Desktop\data.xlsx',tan_loss(1,:),6,strcat('B',num2str(m+1),:,'Q',num2str(m+1)))
% end
% 
% for m=1:length(w)    
% plot(T,tan_loss(1,:))
% hold on
% end
% hold off
% figure
% 
% 
% 
% for j=1:length(T)
% for m=1:length(w) 
% bbb(j)=tan_loss(1,j);
% end
% end
% 
% for j=1:length(T)
% semilogx(w/2/pi,bbb(j,:))
% hold on;
% end 
% hold off
% figure
% xlswrite('C:\Users\ning\Desktop\data.xlsx',[E'],2,'A2:A17')
% xlswrite('C:\Users\ning\Desktop\data.xlsx',[T],2,'B1:Z1')
% xlswrite('C:\Users\ning\Desktop\data.xlsx',real(epsilon_f(:,:,1)),2,'B2:Z17')
% 
% plot(E,real(epsilon_f(:,:,1)))
% figure
% plot(E,nr_f(:,:,1))
% figure
% 
% xlswrite('C:\Users\ning\Desktop\data.xlsx',[E],3,'B1:Z1')
% xlswrite('C:\Users\ning\Desktop\data.xlsx',[w/2/pi]',3,'A2:A17')
% for m=1:length(w)
% xlswrite('C:\Users\ning\Desktop\data.xlsx',real(epsilon_f(:,1))',3,strcat('B',num2str(m+1),:,'z',num2str(m+1)))
% end
% 
% for m=1:length(w)
% plot(E,real(epsilon_f(:,1)))
% hold on
% end
% hold off
% figure
% 
% xlswrite('C:\Users\ning\Desktop\data.xlsx',[E'],4,'A2:A17')
% xlswrite('C:\Users\ning\Desktop\data.xlsx',[T],4,'B1:Z1')
% xlswrite('C:\Users\ning\Desktop\data.xlsx',nr_f(:,:,1),4,'B2:Z17')
% 
% xlswrite('C:\Users\ning\Desktop\data.xlsx',[E],5,'B1:Z1')
% xlswrite('C:\Users\ning\Desktop\data.xlsx',[w/2/pi]',5,'A2:A17')
% for m=1:length(w)
% xlswrite('C:\Users\ning\Desktop\data.xlsx',nr_f(:,1)',5,strcat('B',num2str(m+1),:,'z',num2str(m+1)))
% end
% 
% for m=1:length(w)
% plot(E,nr_f(:,1))
% hold on
% end
% hold off
% semilogx(w/2/pi,tan_loss(:,:))
% figure
% semilogx(w/2/pi,abs(imag(loss1(:,:))),'r',w/2/pi,abs(imag(loss2(:,:))),'bl',w/2/pi,abs(imag(loss3(:,:))),'b',w/2/pi,abs(imag(loss4(:,:))),'g')
% figure
% semilogx(w/2/pi,nr_f(:,:))
% figure
% plot(tan_loss(:,:),nr_f(:,:))
% figure
% plot(E,real(epsilon_f(:,:)))
% figure
% plot(E,nr_f(:,:))