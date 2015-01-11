function accu = testsvm(test_data, test_label, w, b)
% Test linear SVM 
% Input:
%  test_data: M*D matrix, each row as a sample and each column as a
%  feature
%  test_label: M*1 vector, each row as a label
%  w: feature vector 
%  b: bias term
%
% Output:
%  accu: test accuracy (between [0, 1])
%
% CSCI 576 2014 Fall, Homework 3

values = w'*test_data' + b;

y = zeros(length(values),1);
y(values>=0) = 1;
y(values<0) = -1;

accu = sum(y==test_label)/length(test_label);
% disp(['Test Accuracy = ', num2str(accu)]);