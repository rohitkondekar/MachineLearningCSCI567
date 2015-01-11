function [ output_args ] = nonLinearLIBSVM( train_data, train_label, test_data, test_label )
%NONLINEARLIBSVM Summary of this function goes here
%   Detailed explanation goes here



%Polynomial Kernel
C = [4^-3,4^-2,4^-1,1,4,4^2,4^3,4^4,4^5,4^6,4^7];
degree = [1,2,3];

accuracy = zeros(length(C),length(degree));
time = zeros(length(C),length(degree));

for i=1:length(C)
    for j=1:length(degree)
        startTime = tic;
        options = ['-q -s 0 -t 1 -v 5 -c',' ',num2str(C(i)),' ','-d',' ',num2str(degree(j))];
        accu = svmtrain(train_label,train_data,options);
        endTime = toc(startTime);
        accuracy(i,j) = accu;
        time(i,j) = endTime;
    end
end

for i=1:length(C)
    for j=1:length(degree)
    disp(['Polynomial Kernel: C = ',num2str(C(i)),' degree = ',num2str(degree(j)),' Accuracy = ',num2str(accuracy(i,j)),' Execution time = ' , num2str(time(i,j))]); 
    end
end

[maxAccuracy,I] = max(accuracy(:));
[n,m] = ind2sub(size(accuracy),I);

coefficient = C(n);
deg = degree(m);

disp(' ');
disp(['Total Time taken by Polynomial Kernel = ',num2str(sum(sum(time)))]);

disp(['Polynomial Kernel: C = ',num2str(coefficient),' Degree = ',num2str(deg),' Max Accuracy = ',num2str(maxAccuracy),' Execution Time = ',num2str(time(n,m))]);


options = ['-q -s 0 -t 1 -c',' ',num2str(coefficient),' ','-d',' ',num2str(deg)];
model = svmtrain(train_label,train_data,options);
result = svmpredict(test_label,test_data,model);
test_accuracy = sum(result==test_label)/length(test_label);
disp(['Polynomial Kernel: Test Accuracy using C = ',num2str(coefficient),' and degree = ',num2str(deg),' is ',num2str(test_accuracy*100)]);









%----------------------------
%RBF  Kernel

disp('----------------');
disp('----------------');
disp(' ');
disp('RBF Kernel');


C = [4^-3,4^-2,4^-1,1,4,4^2,4^3,4^4,4^5,4^6,4^7];
gamma = [4^-7,4^-6,4^-5,4^-4,4^-3,4^-2];

accuracy = zeros(length(C),length(gamma));
time = zeros(length(C),length(gamma));

for i=1:length(C)
    for j=1:length(gamma)
        startTime = tic;
        options = ['-q -s 0 -t 2 -v 5 -c',' ',num2str(C(i)),' ','-g',' ',num2str(gamma(j))];
        accu = svmtrain(train_label,train_data,options);
        endTime = toc(startTime);
        accuracy(i,j) = accu;
        time(i,j) = endTime;
    end
end

for i=1:length(C)
    for j=1:length(gamma)
    disp(['RBF Kernel: C = ',num2str(C(i)),' gamma = ',num2str(gamma(j)),' Accuracy = ',num2str(accuracy(i,j)),' Execution time = ' , num2str(time(i,j))]); 
    end
end

[maxAccuracy,I] = max(accuracy(:));
[n,m] = ind2sub(size(accuracy),I);

coefficient = C(n);
gam = gamma(m);



disp(' ');
disp(['Total Time taken by RBF Kernel = ',num2str(sum(sum(time)))]);
disp(['RBF Kernel: C = ',num2str(coefficient),' gamma = ',num2str(gam),' Max Accuracy = ',num2str(maxAccuracy),' Execution Time = ',num2str(time(n,m))]);


options = ['-q -s 0 -t 2 -c',' ',num2str(coefficient),' ','-g',' ',num2str(gam)];
model = svmtrain(train_label,train_data,options);
result = svmpredict(test_label,test_data,model);
test_accuracy = sum(result==test_label)/length(test_label);
disp(['RBF Kernel: Test Accuracy using C = ',num2str(coefficient),' and Gamma = ',num2str(gam),' is ',num2str(test_accuracy*100)]);


end

