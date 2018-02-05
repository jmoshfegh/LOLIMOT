clc;
clear;
close all


Initial=.2*ones(17,1);
x=mackeyglass(1500,0,.2,.1,10,Initial);
for t=124:1123
    AllDataInput(t-123,:)=[x(t-24) x(t-18) x(t-12) x(t-6)];
    AllDataTarget(t-123,1)=[x(t)];
end
clear t x;
input=AllDataInput(1:500,:);
DataTestInput=AllDataInput(501:1000,:);
target=AllDataTarget(1:500,:);
DataTestTarget=AllDataTarget(501:1000,:);


LLM_num=10;
training;