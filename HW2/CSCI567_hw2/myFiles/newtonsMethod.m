function [ output_args ] = newtonsMethod( train_data, train_label, name, initial_weights, initial_bias)
%NEWTONSMETHOD Summary of this function goes here
%   Detailed explanation goes here
warning('off');

% add intercept term to data
train_data = [ones(size(train_data,1), 1), train_data];
% with 1 bias
weights = [initial_bias,initial_weights']';%zeros(size(train_data,2),1);
crsEnt = zeros(50,1);
step = 1;
while step<51
    [crossEntropy, grad , Hessian] = getCrossEntropyUnregularized(weights, train_data,train_label);
    weights = weights - Hessian\grad;
    crsEnt(step) = crossEntropy;
    step = step+1;
end
disp(strcat('Newtons NonRegularized Norm =',num2str(norm(weights,2))));
disp(strcat('Newtons NonRegularized CrossEntropy = ',num2str(crsEnt(50))))

figure
plot(crsEnt', 'linewidth',2, 'color','R');
xlabel('Steps');
ylabel('CrossEntropy');
title(strcat(name,': Newtons Method: No Regularization i.e. Lambda=0')); 


% ----------------------------------------------------------
% With Regularization

lambda = linspace(0,.5,11);
crsEnt = zeros(length(lambda),50);
for lamb=1:length(lambda)
    step = 1;
    weights = [initial_bias,initial_weights']';
    while step<51
        [crossEntropy, grad , Hessian] = getCrossEntropyRegularized(weights, train_data,train_label,lambda(lamb));
        weights = weights - Hessian\grad;
        crsEnt(lamb,step) = crossEntropy;
        step = step+1;
    end
    disp(strcat(name,': Newtons Regularized Norm =',num2str(norm(weights,2)),' lambda = ',num2str(lambda(lamb))));
    disp(strcat(name,':Newtons NonRegularized for lamb = ',num2str(lambda(lamb)),' CrossEntropy = ',num2str(crsEnt(lamb,50))));
end


figure
plot(crsEnt(:,:)','linewidth',2);   
legendCell = cellstr(num2str(lambda', '%-f'));
legend(legendCell);
xlabel('Steps');
ylabel('CrossEntropy');
title(strcat(name,': Newtons Method: With Regularization at different lambdas')); 


figure
plot(crsEnt(2:length(lambda),:)','linewidth',2);
legendCell = cellstr(num2str(lambda(2:length(lambda))', '%-f'));
legend(legendCell);
xlabel('Steps');
ylabel('CrossEntropy');
title(strcat(name,': Newtons Method: With Regularization at different lambdas - lambda=0 not plotted here')); 

% ----------------------------------------------------------

warning('on');
end


function [crossEntropy, grad , Hessian] = getCrossEntropyUnregularized(w, x, y)

hypothesis = sigmoid(x*w);     
crossEntropy = -(y.*log(hypothesis) + (1-y).*log(1-hypothesis));
crossEntropy = sum(crossEntropy);
    
grad = x'* (hypothesis-y);
Hessian = x' * diag(hypothesis) * diag(1-hypothesis) * x;
end


function [crossEntropy, grad , Hessian] = getCrossEntropyRegularized(w, x, y, lambda)

hypothesis = sigmoid(x*w);     
crossEntropy = -(y.*log(hypothesis) + (1-y).*log(1-hypothesis));
crossEntropy = sum(crossEntropy) + lambda*norm(w(2:length(w)))^2;
    
G = 2*lambda.*w; 
G(1) = 0;
H = 2*lambda.*eye(size(x,2)); 
H(1) = 0;
grad = (x' * (hypothesis-y)) + G;
Hessian = (x' * diag(hypothesis) * diag(1-hypothesis) * x) + H;
end


function y = sigmoid(x)
y = 1.0 ./ ( 1.0 + exp(-x) );
for i=1:length(y)
    if y(i)<exp(-16)
        y(i) = exp(-16);
    elseif y(i)>1-exp(-16)
        y(i) = 1-exp(-16);
    end
end
end

