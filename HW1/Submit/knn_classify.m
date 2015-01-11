function [new_accu, train_accu] = knn_classify(train_data, train_label, new_data, new_label, k)
% k-nearest neighbor classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%  k: number of nearest neighbors
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data (using leave-one-out
%  strategy)
%
% CSCI 576 2014 Fall, Homework 1

% Data Assumed to come in Binary Format for features : 21 features:
% Distance used is Hamming Distance

    % inner function for knn
    % isLOO is used for getting training accuracy - to ignore that row
    function accuracy = knn(train_data, train_label, new_data, new_label, isLOO, isHamming)       

        resultClass = zeros(length(new_data),1);
        
        for testRow = 1:length(new_data)
            distanceMatrix = zeros(length(train_data),2);
            testMat = repmat(new_data(testRow,:),length(train_data),1); 
            
            
            % ------------------------------------------
            % Mode Hamming Distance - Specific to data set
            if isHamming
                testMat = abs(train_data-testMat); % Hamming Distance Calculation

                for distRow = 1:length(testMat)
                    distance = 0;
                     tmp = 0;
                     for ind=1:12            
                         tmp = tmp + testMat(distRow,ind);
                     end
                     distance = distance + tmp/4;

                     tmp = 0;
                     for ind=13:21            
                         tmp = tmp + testMat(distRow,ind);
                     end
                     distance = distance + tmp/3;      
                     if isLOO && distRow==testRow
                          distanceMatrix(distRow,1) = 100000000;
                     else
                         distanceMatrix(distRow,1) = distance;
                     end                
                     distanceMatrix(distRow,2) = train_label(distRow);
                end
            % ------------------------------------------
            else
                % ------------------------------------------
                % Mode Euclidian Distance - Non-Specific to data set           
                for distRow = 1:length(testMat)
                    if isLOO && distRow==testRow
                          distanceMatrix(distRow,1) = 100000000;
                    else
                         distanceMatrix(distRow,1) = norm(testMat(distRow,:) - train_data(distRow,:));
                    end  
                    distanceMatrix(distRow,2) = train_label(distRow);
                end
            end
            
            distanceMatrix = sortrows(distanceMatrix,1);
            classValues = zeros(1,4); 
            for index = 1:k 
               classValues(distanceMatrix(index,2))=classValues(distanceMatrix(index,2))+1;
            end
            
            [V,I] = max(classValues);
            resultClass(testRow) = I;
        end
           accuracy = getAccuracy(resultClass,new_label); 
    end

      train_accu = knn(train_data,train_label,train_data,train_label,1,0);
      new_accu = knn(train_data,train_label,new_data,new_label,0,0);
    
end


function accuracy = getAccuracy(resultClass, newLabel)
accuracy = (sum(resultClass==newLabel)/length(resultClass));
accuracy = accuracy*100;
end


% % Creates Dummy variable and concatenates it to the table
% function table = createDummyConcat(data,s,l)
% for i = s:l
%     data = horzcat(data,num2cell(dummyvar(categorical(data(:,i)))));
% end
% table = data;
% end






