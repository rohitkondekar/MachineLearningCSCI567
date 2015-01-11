function [new_accu, train_accu] = naive_bayes(train_data, train_label, new_data, new_label)
% naive bayes classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data 
%
% CSCI 576 2014 Fall, Homework 1


%Global Constants

% to make it generic: automate it to unique(train_label)
numClass = 4;

% calculate Prior probabilities for different classes

prior = zeros(1,numClass);
for i=1:numClass
    val = sum(train_label == i)/length(train_label);
    if val==0
       val = 0.0000000001 ;
    end
    prior(i) = log(val);
end

% calculate conditional probabilities for all features
cond_probability = zeros(21:2:4);
for i = 1:size(train_data,2)
    for j = 0:1
        for k = 1:4       
            val = sum((train_data(:,i)==j) & (train_label==k))/sum(train_label==k);
            if val==0
                val = 0.0000000001 ;
            end
            cond_probability(i,j+1,k) = log(val);
        end
    end
end

%Inner Function - to calculate class
    function labels = calClass(new_data)
        resultClass = zeros(1,length(new_data));
        for row = 1:length(new_data)
           class_prob = zeros(1,numClass);
           for col = 1:size(new_data,2)
               %prob for classes
                for class = 1:numClass
                    if new_data(row,col) ~= 0
                        class_prob(class) = class_prob(class)+cond_probability(col,new_data(row,col)+1,class);
                    end
                end
           end
           
           for z=1:numClass
               class_prob(z) = class_prob(z) + prior(z);
           end
           
           [V,I] = max(class_prob);
           resultClass(row) = I;
        end
        labels = resultClass';
    end

%Testing Accuracy
resultClass = calClass(new_data);
new_accu = getAccuracy(resultClass,new_label);

%Training Accuracy
resultClass = calClass(train_data);
train_accu = getAccuracy(resultClass,train_label);

end

function accuracy = getAccuracy(resultClass, newLabel)
accuracy = (sum(resultClass==newLabel)/length(resultClass))*100;
end


