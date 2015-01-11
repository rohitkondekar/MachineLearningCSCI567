function [ required_weights, required_bias ] = batchGradientDescent( train_data, train_label, name, linestyles, Markers)
% BATCH GRADIENT DESCENT 
% Summary of this function goes here
% Detailed explanation goes here

eta = [0.001, 0.01, 0.05, 0.1, 0.5]; %step size


% % Unregularized Part ----------------

crsEnt = zeros(length(eta),50);
disp('L2 norm for different step size -- ');
for e=1:length(eta)
    step = 0;
    weights = zeros(size(train_data,2),1);
    b = 0.1;
    while step<51
        [crossEntropy, gradCrossEntropy, gradCrossB] = getCrossEntropyUnregularized(b,weights,train_data,train_label);
         weights = weights + eta(e).*(gradCrossEntropy'); % this is + because I am doing y-hypothesis
         b = b + eta(e)*(sum(gradCrossB));
        step = step+1;
        crsEnt(e,step) = crossEntropy;
    end
    disp([eta(e),norm(weights,2)]);
end
% 
% 
% % -------- Graph Uncomment---------
% 
MarkerEdgeColors=hsv(size(crsEnt,1));  % n is the number of different items you have


clear plot;
figure
hold on;
for i=1:size(crsEnt,1)
    plot(crsEnt(i,:)',[linestyles{i} Markers(i)],'Color',MarkerEdgeColors(i,:),'linewidth',2);
end
%plot(crsEnt(:,:)','linewidth',2,MyStyles);
legendCell = cellstr(num2str(eta', 'n=%-f'));
legend(legendCell);
xlabel('Steps');
ylabel('CrossEntropy');
title(strcat(name,': Without Regularization at Different Step Sizes'));
hold off;
% 
% % ---------------------------------
% 
% 
% % End Unregularized Part ----------------



% Regularized Part Lambda - Steps----------------
disp('L2 norm for different lambda values in case of Regularized Linear Regression -- ');
lambda = linspace(0,.5,11); %break 0 - 0.5 into 11 parts
crsEnt = zeros(length(eta),50);
for lamb=1:length(lambda)
    for e=1:length(eta)
        step = 0;
        weights = zeros(size(train_data,2),1);
        b = 0;
        while step<51
            [crossEntropy, gradCrossEntropy, gradCrossB] = getCrossEntropyRegularized(b,weights,train_data,train_label,lambda(lamb));
             weights = weights + eta(e)*(gradCrossEntropy');
             b = b + eta(e)*(sum(gradCrossB));
            step = step+1;
            crsEnt(e,step) = crossEntropy;
        end
        
        if(eta(e)==0.001)
            disp([lambda(lamb),norm(weights,2)]);
        end
        
        if(lamb == 2 && eta(e)==0.01)            
            required_weights = weights;
            required_bias = b;
        end
    end
    
    
    
    % -------- Graph Uncomment---------
    if lamb == 3 %0.1
        MarkerEdgeColors =hsv(size(crsEnt,1));  % n is the number of different items you have
        figure
        hold on
        for i=1:size(crsEnt,1)
            plot(crsEnt(i,:)',[linestyles{i} Markers(i)],'Color',MarkerEdgeColors(i,:),'linewidth',2);
        end

%         figure
%         plot(crsEnt(1:4,:)');
        legendCell = cellstr(num2str(eta', 'n=%-f'));
        legend(legendCell);
        xlabel('Steps');
        ylabel('CrossEntropy');
        title(strcat(name,': With Regularization at Different Step Sizes for Lambda = ',num2str(lambda(lamb)))); 
    end
    % ---------------------------------
end
% % End of Regularized Part ----------------



% Regularized Part  Steps -- Lambda----------------
% lambda = linspace(0,.5,11);
% crsEnt = zeros(length(lambda),1);
% for e=1:length(eta)
%     for lamb=1:length(lambda)
%         step = 0;
%         weights = zeros(size(train_data,2),1);
%         b = 0.1;
%         while step<51
%             [crossEntropy, gradCrossEntropy, gradCrossB] = getCrossEntropyRegularized(b,weights,train_data,train_label,lambda(lamb));
%              weights = weights + eta(e)*(gradCrossEntropy');
%              b = b + eta(e)*(sum(gradCrossB));
%             step = step+1;            
%         end
%         crsEnt(lamb) = crossEntropy;
%     end
%     
%     % -------- Graph Uncomment---------
%     figure
% %         line(lambda,crsEnt);
%         plot(lambda,crsEnt,'-o');
% %     for i=1:length(lambda)
% %         line(lambda(i),crsEnt(i));
% %     end
% %     figure
% %     plot(crsEnt(:,:)','linewidth',2 );
%     xlabel('Regularization Coefficient');
%     ylabel('CrossEntropy');
%     title(strcat(name,': With Regularization at different Regularization Coefficients for step size as ',num2str(eta(e)))); 
% %     
    % ---------------------------------

% end
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
