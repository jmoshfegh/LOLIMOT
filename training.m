% clear;
% close all
% laser;
% yearKP;
%p=2;                              %dimention of input
M=1;   
% LLM_num=12;
%number of LMM
zarib=.33;                         %standard deviation per extention of the hyperrectangle
z=1;                               %The max number of LLM
input;                                 %input----input=[]N*p
target;                         %desired output----output=[]N*1
%N=441;                            %The number of data sample
a=min(input);                          %The initial of intervals----a=[]M*p(in general)
b=max(input);                          %The end of intervals----b=[]M*p(in general)
c=(a+b)/2;                         %The center of gausian functions----c=[]M*p
sigma=(b-a)*zarib;                 %The deviation of gausian functions----sigma=[]M*p
[N p]=size(input);
X=[ones(N,1) input];                   %input vector for network----X=[]N*(p+1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%computing membership MSFs:
input_expand=repmat(input',[1 1 M]);                   %generating power----input_expand=[]p*N*M
c_expand=permute(repmat(c,[1 1 N]),[2 3 1]);           %The center of gausian functions----c_expand=[]p*N*M
sigma_expand=permute(repmat(sigma,[1 1 N]),[2 3 1]);      %The deviation of gausian functions----sigma_expand=[]p*N*M
power=((input_expand-c_expand)./sigma_expand).^2;               %generating power----power=[]p*N*M
if p==1
    powered=reshape(powered,N,M);                     %----[]N*M
    MSF_temp=exp(-1/2*powered');                      %the membership functions----MSF=[]M*N
else
    powered=sum(power);                       %----[]1*N*M
    powered=reshape(powered,N,M);                     %----[]N*M
    MSF=exp(-1/2*powered');                      %the membership functions----MSF=[]M*N
end
%End of computing MSFs:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%computing validity function "phi"(normalized MSFs):
summation=sum(MSF);                                 %[]1*N
RepSummation=repmat(summation,M,1);                   %[]M*N
phi=MSF./RepSummation;                              %the validity function----phi=[]M*N
%end of computing function "phi"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:M
    Q=diag(phi(i,:));                          %----Q=[]N*N
    w(:,i)=inv(X'*Q*X)*X'*Q*target;         %waighting matrix----w=[](p+1)*M
    output(:,i)=X*w(:,i);                           %output of LLM i----output=[]N*M
    e(:,i)=target-output(:,i);                   %error function----e=[]N*M
    I(i,1)=phi(i,:)*(e(:,i).^2);                    %The loss function----I=[]M*1
end
%Computing Network Output and Error
NetworkOutput=(output'.*phi)';
NetworkError=target-NetworkOutput;
TotalError=NetworkError'*NetworkError;
MSE_Training(1,1)=TotalError/(N);
%End of Computing Network Output and Error
%figure;
%hold;
%plot(target,'b');
%plot(NetworkOutput,'r');
testing;
%figure;
%hold;
%plot(MSE_Training,'b.');
%plot(MSE_Testing,'r.');
%pause
[r s]=max(I);                              %r=the max of least square----s=The worst LLM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for z=2:LLM_num
    %The begining of dividing worst LLM
    M=M+1;
    disp(['Level of LLM = ' num2str(M)]);
    for j=1:p
        %Changing in intervals:
        b_temp=b;
        b_temp(M,:)=b_temp(s,:);
        b_temp(s,j)=(a(s,j)+b(s,j))/2;
        a_temp=a;
        a_temp(M,:)=a_temp(s,:);
        a_temp(M,j)=(a(s,j)+b(s,j))/2;
        %The end of updating intervals.
        c_temp=(a_temp+b_temp)/2;
        sigma_temp=(b_temp-a_temp)*zarib;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %computing "MSF_temp"
        clear MSF_temp;
        input_expand=repmat(input',[1 1 M]);                   %generating power----input_expand=[]p*N*M
        c_expand=permute(repmat(c_temp,[1 1 N]),[2 3 1]);           %The center of gausian functions----c_expand=[]p*N*M
        sigma_expand=permute(repmat(sigma_temp,[1 1 N]),[2 3 1]);      %The deviation of gausian functions----sigma_expand=[]p*N*M
        power=((input_expand-c_expand)./sigma_expand).^2;               %generating power----power=[]p*N*M
        if p==1
            powered=reshape(powered,N,M);                     %----[]N*M
            MSF_temp=exp(-1/2*powered');                      %the membership functions----MSF=[]M*N
        else
            powered=sum(power);                       %----[]1*N*M
            powered=reshape(powered,N,M);                     %----[]N*M
            MSF_temp=exp(-1/2*powered');                      %the membership functions----MSF=[]M*N
        end
        %end of computing of "MSF_temp"
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %computing validity function "phi_temp"(normalized MSFs):
        summation_temp=sum(MSF_temp);                                 %[]1*N
        RepSummation_temp=repmat(summation_temp,M,1);                   %[]M*N
        phi_temp=MSF_temp./RepSummation_temp;                              %the validity function----phi=[]M*N
        %end of computing function "phi"
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i=1:M
            Q=diag(phi_temp(i,:));                          %----Q=[]N*N
            w(:,i)=inv(X'*Q*X)*X'*Q*target;         %waighting matrix----w=[](p+1)*M
            output_temp(:,i)=X*w(:,i);                           %output of LLM i----output=[]N*M
            e_temp(:,i)=target-output_temp(:,i);                   %error function----e=[]N*M
            I_temp(i,1)=phi_temp(i,:)*(e_temp(:,i).^2);                    %The loss function----I=[]M*1
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Computing Network Output and Error
        NetworkOutput_temp=sum(output_temp'.*phi_temp)';
        NetworkError_temp=target-NetworkOutput_temp;
        TotalError_temp(j,1)=NetworkError_temp'*NetworkError_temp;
        %End of Computing Network Output and Error
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    [f g]=min(TotalError_temp);                       %f=The min of squared error in all division----g=The dimension that best division occur along it
    %Updating the model:
        %Changing in intervals:
        b_temp=b;
        b_temp(M,:)=b_temp(s,:);
        b_temp(s,g)=(a(s,g)+b(s,g))/2;
        a_temp=a;
        a_temp(M,:)=a_temp(s,:);
        a_temp(M,g)=(a(s,g)+b(s,g))/2;
        %The end of updating intervals.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %computing "MSF_temp"
        clear MSF_temp;
        input_expand=repmat(input',[1 1 M]);                   %generating power----input_expand=[]p*N*M
        c_expand=permute(repmat(c_temp,[1 1 N]),[2 3 1]);           %The center of gausian functions----c_expand=[]p*N*M
        sigma_expand=permute(repmat(sigma_temp,[1 1 N]),[2 3 1]);      %The deviation of gausian functions----sigma_expand=[]p*N*M
        power=((input_expand-c_expand)./sigma_expand).^2;               %generating power----power=[]p*N*M
        if p==1
            powered=reshape(powered,N,M);                     %----[]N*M
            MSF_temp=exp(-1/2*powered');                      %the membership functions----MSF=[]M*N
        else
            powered=sum(power);                       %----[]1*N*M
            powered=reshape(powered,N,M);                     %----[]N*M
            MSF_temp=exp(-1/2*powered');                      %the membership functions----MSF=[]M*N
        end
        %end of computing of "MSF_temp"
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %computing validity function "phi_temp"(normalized MSFs):
        summation=sum(MSF_temp);                                 %[]1*N
        RepSummation=repmat(summation,M,1);                   %[]M*N
        phi_temp=MSF_temp./RepSummation;                              %the validity function----phi=[]M*N
        %end of computing function "phi"
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i=1:M
            Q=diag(phi_temp(i,:));                          %----Q=[]N*N
            w(:,i)=inv(X'*Q*X)*X'*Q*target;         %waighting matrix----w=[](p+1)*M
            output(:,i)=X*w(:,i);                           %output of LLM i----output=[]N*M
            e(:,i)=target-output(:,i);                   %error function----e=[]N*M
            I(i,1)=phi_temp(i,:)*(e(:,i).^2);                    %The loss function----I=[]M*1
        end
        phi=phi_temp;
        MSF=MSF_temp;
        a=a_temp;
        b=b_temp;
        c=c_temp;
        sigma=sigma_temp;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Computing Network Output and Error
    NetworkOutput=sum(output'.*phi)';
    NetworkError=target-NetworkOutput;
    TotalError=NetworkError'*NetworkError;
    MSE_Training(z,1)=TotalError/(N);
    %End of Computing Network Output and Error
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    testing;
    if (M == LLM_num)
        figure;
        subplot(2,1,1);
        box on;
        plot(1:N,target,'b',1:N,NetworkOutput,'g');
        legend('Target','Network Output','Location','EastOutside');

        subplot(2,1,2);
        box on;
        plot(1:N_test,DataTestTarget,'b',1:N_test,NetworkOutputTest,'g');
        legend('Observed Values','Modeled Values','Location','EastOutside');
    end

    [r s]=max(I);                              %r=the max of least square----s=The worst LLM
    clear I;
end

[optimum,i]=min(MSE_Testing);
disp(['Optimum Number of LLM = ' num2str(i)]);

MSE_Training;
MSE_Testing;
%%
figure;
plot(1:LLM_num,MSE_Training,'s',1:LLM_num,MSE_Testing,'^');
legend('MSE for Training Data','MSE for Testing Data','Location', 'NorthEast');
box on;
axis([0 LLM_num+1 0 MSE_Testing(1)]);

figure;
box on;
plot([min(DataTestTarget) max(DataTestTarget)],[min(DataTestTarget) max(DataTestTarget)],'b');
hold on;
plot(DataTestTarget,NetworkOutputTest,'ro');
xlabel('Observed Values');
ylabel('Modeled Values');
