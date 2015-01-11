function [ output_args ] = libSVM( train_data, train_label, test_data, test_label )
%LIBSVM Summary of this function goes here
%   Detailed explanation goes here
% Linear SVM
 

C = [4^-6,4^-5,4^-4,4^-3,4^-2,4^-1,1,4,4^2];

accuracy = zeros(length(C),1);
time = zeros(length(C),1);

for j=1:length(C)
    startTime = tic;
    options = ['-q -s 0 -t 0 -v 5 -c',' ',num2str(C(j))];
    accu = svmtrain(train_label,train_data,options);
    endTime = toc(startTime);
    accuracy(j) = accu;
    time(j) = endTime;
end

[Y I] = max(accuracy);
coefficient = C(I);

accuracy
time

disp(['C = ',num2str(coefficient),' Max Accuracy = ',num2str(Y),' Execution Time = ',num2str(time(I))]);

options = ['-q -s 0 -t 0 -c',' ',num2str(coefficient)];
model = svmtrain(train_label,train_data,options);
result = svmpredict(test_label,test_data,model);
test_accuracy = sum(result==test_label)/length(test_label);
disp(['Test Accuracy using C = ',num2str(coefficient),' is ',num2str(test_accuracy*100)]);


end

