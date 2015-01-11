function [ ] = CSCI567_hw3( w , b )
%CSCI567_HW3 Summary of this function goes here
%   Detailed explanation goes here
path = pwd;
addpath(genpath(strcat(path,'/myFiles')));
addpath(genpath(strcat(path,'/hw3_data')));

load('splice_train.mat');
train_label = label;
train_data = data;
load('splice_test.mat');
test_label = label;
test_data = data;

%Notice that the mean and standard deviation should be estimated from the training data and then applied to both datasets.
mean_training_data = mean(train_data);
std_training_data = std(train_data);

%Normalize Data
train_data = bsxfun(@rdivide,bsxfun(@minus,train_data,mean_training_data),std_training_data);
test_data = bsxfun(@rdivide,bsxfun(@minus,test_data,mean_training_data),std_training_data);

% [w,b] = trainsvm(train_data,train_label,16);
% testsvm(test_data, test_label, w, b);


[ accuracy, time, C ] = crossValidation(train_data,train_label,test_data,test_label);
[ w, b] = trainsvm(train_data,train_label,C);
test_accuracy = testsvm(test_data, test_label, w, b);
disp(['Test Accuracy = ', num2str(test_accuracy*100)]);

disp(' ');
disp(' ');
disp('Using LibSVM now ---------');
libSVM(train_data,train_label,test_data, test_label);

disp(' ');
disp(' ');
disp('Using NonLinear Polynomial LibSVM now ---------');
nonLinearLIBSVM(train_data,train_label,test_data, test_label)

end

