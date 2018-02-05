clc;
clear;
close all

load('yearssn.dat');
x=yearssn(:,2);
for t=7:size(yearssn,1)
    AllDataInput(t-6,:)=[x(t-6) x(t-5) x(t-4) x(t-3) x(t-2) x(t-1)];
    AllDataTarget(t-6,1)=[x(t)];
end
clear t x;
input=AllDataInput(1:200,:);
DataTestInput=AllDataInput(201:295,:);
target=AllDataTarget(1:200,:);
DataTestTarget=AllDataTarget(201:295,:);

LLM_num=5;
training;