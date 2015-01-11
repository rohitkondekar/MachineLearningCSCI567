function [ output_args ] = crossEntropyVSRegu( train_data, train_label, test_data, test_label, name)
%CROSSENTROPYVSREGU Summary of this function goes here
%   Detailed explanation goes here

eta = [0.001, 0.01, 0.05, 0.1, 0.5]; %step size

% Regularized Part  Steps -- Lambda----------------
lambda = linspace(0,.5,11);
crsEnt = zeros(length(lambda),1);
crsEntTest = zeros(length(lambda),1);

for e=1:length(eta)
    for lamb=1:length(lambda)
        step = 0;
        weights = zeros(size(train_data,2),1);
        b = 0.1;
        while step<51
            [crossEntropy, gradCrossEntropy, gradCrossB] = getCrossEntropyRegularized(b,weights,train_data,train_label,lambda(lamb));
             weights = weights + eta(e)*(gradCrossEntropy');
             b = b + eta(e)*(sum(gradCrossB));
            step = step+1;            
        end
        crsEnt(lamb) = crossEntropy;
        
        %Test
        step = 0;
        weights = zeros(size(test_data,2),1);
        b = 0.1;
        while step<51
            [crossEntropy, gradCrossEntropy, gradCrossB] = getCrossEntropyRegularized(b,weights,test_data,test_label,lambda(lamb));
             weights = weights + eta(e)*(gradCrossEntropy');
             b = b + eta(e)*(sum(gradCrossB));
            step = step+1;            
        end        
        crsEntTest(lamb) = crossEntropy;
    end
    
    % -------- Graph Uncomment---------
    figure
    plot(lambda,crsEntTest,'-x','linewidth',2,'color','b');
    hold on;
    plot(lambda,crsEnt,'-o','linewidth',2,'color','r');
    hold off;
    legend('Test','Train');
    xlabel('Regularization Coefficient');
    ylabel('CrossEntropy');
    title(strcat(name, ': With Regularization at different Regularization Coefficients for step size as ',num2str(eta(e)))); 
    % ---------------------------------
end
end


function [crossEntropy, gradCrossEntropy, gradCrossB] = getCrossEntropyRegularized(b,w, x, y, lambda)
    hypothesis = sigmoid(b+x*w);        
    crossEntropy = -(y.*log(hypothesis) + (1-y).*log(1-hypothesis));
    crossEntropy = sum(crossEntropy) + lambda*(w'*w);
    gradCrossB = (y-hypothesis);
    gradCrossEntropy = (gradCrossB'*x) - 2*lambda*w';
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


