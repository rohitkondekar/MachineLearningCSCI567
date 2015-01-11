function [ train_accuracy, test_accuracy ] = logit( train_data, train_label, new_data, new_label )
%LOGIT Summary of this function goes here
%   Detailed explanation goes here


% creates dummy variables and concatenates it to the matrix
        train_data = createDummyConcat(train_data,1,6);
        new_data = createDummyConcat(new_data,1,6);                 
        t = mnrfit(train_data,double(categorical(train_label)));       
        result = mnrval(t,new_data);
        
        resultClass = cell(length(new_data),1);
        
        for i=1:length(result)
            if result(i,1)>result(i,2) && result(i,1)>result(i,3) && result(i,1)>result(i,4)
                 resultClass{i} = 'acc';
            elseif result(i,2)>result(i,1) && result(i,2)>result(i,3) && result(i,2)>result(i,4)
                resultClass{i} = 'good';
            elseif result(i,3)>result(i,1) && result(i,3)>result(i,2) && result(i,3)>result(i,4)
                resultClass{i} = 'unacc';
            else
                resultClass{i} = 'vgood';
            end           
        end
        
        disp(getAccuracy(resultClass,new_label)*100);
        
        
        
        
        result = mnrval(t,train_data);        
        resultClass = cell(length(train_data),1);
        
        for i=1:length(result)
            if result(i,1)>result(i,2) && result(i,1)>result(i,3) && result(i,1)>result(i,4)
                 resultClass{i} = 'acc';
            elseif result(i,2)>result(i,1) && result(i,2)>result(i,3) && result(i,2)>result(i,4)
                resultClass{i} = 'good';
            elseif result(i,3)>result(i,1) && result(i,3)>result(i,2) && result(i,3)>result(i,4)
                resultClass{i} = 'unacc';
            else
                resultClass{i} = 'vgood';
            end           
        end        
        disp(getAccuracy(resultClass,train_label)*100);
        
        
        
        
end

% Creates Dummy variable and concatenates it to the table
% return double matrix
function table = createDummyConcat(data,s,l)

newdata = dummyvar(categorical(data(:,s)));
for i = s+1:l
    newdata = horzcat(newdata,(dummyvar(categorical(data(:,i)))));
end
table = newdata;
end


function accuracy = getAccuracy(resultClass, newLabel)
sum = 0;
for index = 1:length(resultClass)
    if (strcmp(resultClass{index},newLabel{index}))
        sum = sum+1;
    end
end
accuracy = sum/length(resultClass);
end
