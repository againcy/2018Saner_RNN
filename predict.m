function [p, Y]=predict(U, V, W, b,c,X)
%PREDICT Predict the label of an input given a trained neural network

m=size(X,1);
input_layer_size=size(U,2);
hidden_layer_size=size(U,1);
num_labels=size(c,1);
p=zeros(m,1);
Y=zeros(m,1);%target
for i=1:m
    T=X(i,1);
    %-----obtain 1 sample from dataTest X-----
    X_this_sample=zeros(T, input_layer_size);
    for t=1:T
        X_this_sample(t,:)=X(i,2+(input_layer_size+1)*(t-1):(input_layer_size+1)*t);
    end
    Y(i)=X(i,1+(input_layer_size+1)*T);
    s=zeros(hidden_layer_size,T);
    o=zeros(num_labels,T);
    pred=zeros(num_labels,T);
    for t=1:T%Forward
        if t==1
            s(:,1)=tanh(U*(X_this_sample(t,:)')+b);%hidden layer states are stored by column
            o(:,1)=V*s(:,1)+c;
            pred(:,1)=sigmoid(o(:,1));
        else
            s(:,t)=tanh(U*(X_this_sample(t,:)')+W*s(:,t-1)+b);
            o(:,t)=V*s(:,t)+c;
            pred(:,t)=sigmoid(o(:,t));
        end
    end
    p(i)=pred(:,T);
end
Y=~(Y==0);%1 means bug(s)
