function [ output_args ] = q4_c( train_data, train_label, test_data, test_label )
%Q4_C Summary of this function goes here
%   Detailed explanation goes here

    function z = calc&Plot(data, data_label)
        eta = [0.001, 0.01, 0.05, 0.1, 0.5];
        crsEnt = zeros(length(eta),50);
        
    end



end

function [crossEntropy, gradCrossEntropy, gradCrossB] = getCrossEntropyUnregularized(b, w, x, y)
    hypothesis = sigmoid(b+x*w);     
    crossEntropy = -(y.*log(hypothesis) + (1-y).*log(1-hypothesis));
    crossEntropy = sum(crossEntropy);
    
    gradCrossB = (y-hypothesis);
    gradCrossEntropy = (gradCrossB'*x);
end

function [crossEntropy, gradCrossEntropy, gradCrossB] = getCrossEntropyRegularized(b,w, x, y, lambda)
    hypothesis = sigmoid(b+x*w);        
    crossEntropy = -(y.*log(hypothesis) + (1-y).*log(1-hypothesis));
 %disp([hypothesis,b+x*w]);
    crossEntropy = sum(crossEntropy) + lambda*(w'*w);
    
    gradCrossB = (y-hypothesis);
    gradCrossEntropy = (gradCrossB'*x) + lambda*w';
end

function y = sigmoid(x)
y = 1.0 ./ ( 1.0 + exp(-x) );
for i=1:length(y)
    if y(i)<exp(-16)
        y(i) = exp(-16);
    elseif y(i)>0.999
        y(i) = 1-exp(-9);
    end
end
end
