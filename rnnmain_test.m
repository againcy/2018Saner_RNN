% RNN model for defect prediction
% Author: Jianbo Guo 
% Email: jianboguo@outlook.com
% Modified by: Yibin Liu
% Email: yibinliu93@foxmail.com

clear; clc; close all;

%setting parameters
input_layer_size=20;  % size of metrics in HVSM (20 for HVSM built with code metrics and 24 for HVSM built with both code and process metrics)
hidden_layer_size=5; 
iter = 1300;
lambda = 16;
num_labels=1;%2-classification problem

%name of dataset should be 'projectName_train.csv' for training set, and 'projectName_test.csv' for test set.
dataDirectory = 'G:\RNN_Matlab\promise\';%the root directory of datasets
name = 'ant_1.6_1.7';%projectName
dataTest=importdata([dataDirectory, name, '_test.csv']);
dataTrain=importdata([dataDirectory, name, '_train.csv']);

path='G:\RNN_Matlab\Result\test\';%the root directory of results
mkdir(path);
    
fprintf('Doing data = %s \n',name);
%% -------------------rand--------------------------
randTimes = 10; % # of rand, the final result should be the average of the 10 results
for randid=1:randTimes 
    
U=randInitializeWeights(input_layer_size-1,hidden_layer_size);
b=zeros(hidden_layer_size,1);
W=randInitializeWeights(hidden_layer_size-1,hidden_layer_size);
V=randInitializeWeights(hidden_layer_size-1,num_labels);
c=zeros(num_labels,1);
params=[U(:);V(:);W(:); b(:); c(:)];

% fprintf('\n Training reccurent neural network...\n');
options = optimset('MaxIter', iter); % # of interations

costFunc=@(p) rnnCostFunction(p, input_layer_size, hidden_layer_size, num_labels,...
                                  dataTrain, lambda);
% fprintf('Rand %d\n',i);                              
[rnn_params, cost] = fmincg(costFunc, params, options);

U_size=hidden_layer_size*input_layer_size;
V_size=num_labels*hidden_layer_size;
W_size=hidden_layer_size*hidden_layer_size;
U=reshape(rnn_params(1:U_size), hidden_layer_size,input_layer_size);
V=reshape(rnn_params(U_size+1:U_size+V_size), num_labels, hidden_layer_size);
W=reshape(rnn_params(U_size+V_size+1:U_size+V_size+W_size), hidden_layer_size, hidden_layer_size);
b=reshape(rnn_params(U_size+V_size+W_size+1:U_size+V_size+W_size+hidden_layer_size), hidden_layer_size,1);
c=reshape(rnn_params(U_size+V_size+W_size+hidden_layer_size+1:end), num_labels,1);

[pred, yTest]=predict(U,V,W,b,c,dataTest); 

%output
outputName = [path,'_cm_',name,'_',num2str(randid),'.csv'];%name of the result file 'projectName_randid.csv'
fid = fopen(outputName,'w');
fprintf(fid,'%s\n','test,condition');
fclose(fid);
dlmwrite(outputName,[pred,yTest],'-append');
end

fprintf('Finish doing data = %s \n',name');

