function [ accuracy, time, coefficient ] = crossValidation( train_data, train_label, test_data, test_label)

K = 5; % 5fold cross validation

indices = crossvalind('Kfold', size(train_data,1), K);
C = [4^-6,4^-5,4^-4,4^-3,4^-2,4^-1,1,4,4^2];

accuracy = zeros(length(C),1);
time = zeros(length(C),1);

for j=1:length(C)
    accu = 0;
    t = 0;
    for i=1:K
        bool = indices~=i;
        startTime = tic;
        [w,b] = trainsvm(train_data(bool,:),train_label(bool),C(j));        
        t = t + toc(startTime);
        accu = accu + testsvm(train_data(~bool,:),train_label(~bool),w,b);
    end
    accuracy(j) = accu/K;
    time(j) = t/K;
end

[Y,I] = max(accuracy);
coefficient = C(I);
accuracy
time

disp(['C = ',num2str(coefficient),' Max Accuracy = ',num2str(Y*100),' Execution Time = ',num2str(time(I))]);

[w,b] = trainsvm(train_data,train_label,coefficient);        
test_accuracy = testsvm(test_data,test_label,w,b);
disp(['Test Accuracy using C = ',num2str(coefficient),' is ',test_accuracy]);


end

