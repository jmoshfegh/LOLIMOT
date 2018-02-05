clc;
clear;
close all

x=load('datacomp.dat');
for t=5:2000
    AllDataInput(t-4,:)=[x(t-4) x(t-3) x(t-2) x(t-1)];
    AllDataTarget(t-4,1)=[x(t)];
end
clear t x;
input=AllDataInput(1:1600,:);
DataTestInput=AllDataInput(1601:1996,:);
target=AllDataTarget(1:1600,:);
DataTestTarget=AllDataTarget(1601:1996,:);

LLM_num=3;
training;