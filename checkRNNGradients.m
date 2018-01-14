function checkRNNGradients(lambda)
%CHECKTNNGRADIENTS Creates a small neural network to check the
%backpropagation gradients
%   CHECKRNNGRADIENTS(lambda) Creates a small reccurrent neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient). These two gradient computations should
%   result in very similar values.
input_layer_size=3;
hidden_layer_size=6;
num_labels=1;
m=5;
%--------------------------produce some parameters randomly------------------
init_epsilon=sqrt(6/(num_labels+input_layer_size));
U=rand(hidden_layer_size, input_layer_size)*2*init_epsilon-init_epsilon;
W=rand(hidden_layer_size,hidden_layer_size)*2*init_epsilon-init_epsilon;
V=rand(num_labels, hidden_layer_size)*2*init_epsilon-init_epsilon;
b=rand(hidden_layer_size,1)*2*init_epsilon-init_epsilon;
c=rand(num_labels,1)*2*init_epsilon-init_epsilon;
params=[U(:);V(:); W(:); b(:); c(:)];

%-------------------------produce some samples whose time length are 5----------------------------
dataX=debugInitializeWeights(5*m, input_layer_size);
dataY=mod(1:5*m, num_labels+1)';
dataTrain=zeros(m,1+(input_layer_size+1)*5);
for i=1:m
    dataTrain(i,1)=5;
    for t=1:5
        dataTrain(i,2+(input_layer_size+1)*(t-1):(input_layer_size+1)*t)=dataX(5*(i-1)+t,:);
        dataTrain(i,1+(input_layer_size+1)*t)=dataY(5*(i-1)+t);
    end
end

%short hand for cost function
costFunc=@(p) rnnCostFunction(p,input_layer_size, hidden_layer_size, num_labels,...
                                  dataTrain, lambda);
[cost, grad]=costFunc(params);
numgrad=computeNumericalGradient(costFunc, params);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);

fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);

end



