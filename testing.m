%DataTestInput;                   %input for testing the network----[]N_test*p
%DataTestTarget;                   %target for testing the network----[]N_test*1
[N_test p]=size(DataTestInput);
%N_test;                        %the number of data sample for testing
X_test=[ones(N_test,1) DataTestInput];                   %input vector for network----X=[]N_test*(p+1)
OutputTest=X_test*w;                                     %Network output for testing----[]N_test*M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%computing membership function MSFs:
clear MSF_test;
DataTestInput_expand=repmat(DataTestInput',[1 1 M]);           %generating power----input_expand=[]p*N_test*M
c_expand=permute(repmat(c,[1 1 N_test]),[2 3 1]);           %The center of gausian functions----c_expand=[]p*N*M
sigma_expand=permute(repmat(sigma,[1 1 N_test]),[2 3 1]);   %The deviation of gausian functions----sigma_expand=[]p*N*M
power=((DataTestInput_expand-c_expand)./sigma_expand).^2;               %generating power----power=[]p*N*M
if p==1
    powered=reshape(powered,N_test,M);                     %----[]N*M
    MSF_test=exp(-1/2*powered');                      %the membership functions----MSF=[]M*N
else
    powered=sum(power);                       %----[]1*N*M
    powered=reshape(powered,N_test,M);                     %----[]N*M
    MSF_test=exp(-1/2*powered');                      %the membership functions----MSF=[]M*N
end
%End of computing MSFs:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%computing validity function "phi"(normalized MSFs):
summation=sum(MSF_test);                                 %[]1*N_test
RepSummation=repmat(summation,M,1);                   %[]M*N_test
phi_test=MSF_test./RepSummation;                              %the validity function----phi=[]M*N_test
%end of computing function "phi"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NetworkOutputTest=sum(OutputTest'.*phi_test)';                 %----[]N_test*1
NetworkErrorTest=DataTestTarget-NetworkOutputTest;                                %----[]N_test*1
TotalErrorTest=NetworkErrorTest'*NetworkErrorTest;                                 %----[]1*1
MSE_Testing(z,1)=TotalErrorTest/N_test;                               %----[]M*1

