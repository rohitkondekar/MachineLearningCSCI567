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


% Class Values:
%
% unacc, acc, good, vgood


% Attributes:
%
% buying: vhigh, high, med, low.
% maint: vhigh, high, med, low.
% doors: 2, 3, 4, 5more.
% persons: 2, 4, more.
% lug_boot: small, med, big.
% safety: low, med, high.


% converting categorical features to dummy variables

% inner function for knn
% isLOO is used for getting training accuracy - to ignore that row
function accuracy = knn(train_data, train_label, new_data, new_label, isLOO)

train_data = createDummyConcat(train_data,1,6);
new_data = createDummyConcat(new_data,1,6);
resultClass = cell(length(new_data),12);

for testRow = 1:length(new_data)

testMat = repmat(new_data(testRow,7:27),length(train_data),1);
testMat = abs(cell2mat(train_data(:,7:27))-cell2mat(testMat));

distanceMatrix = cell(length(train_data),2);


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
distanceMatrix{distRow,1} = 100000000;
else
distanceMatrix{distRow,1} = distance;
end

distanceMatrix(distRow,2) = train_label(distRow);
end
distanceMatrix = sortrows(distanceMatrix,1);

unacc = 0;
acc = 0;
good = 0;
vgood = 0;

rindex = 1;
for k = 1:23

for index = 1:k
switch distanceMatrix{index,2}
case 'unacc'
unacc = unacc+1;
case 'acc'
acc = acc+1;
case 'good'
good = good+1;
case 'vgood'
vgood = vgood+1;
end
end

if unacc>=acc && unacc>=good && unacc>=vgood
resultClass{testRow,rindex} = 'unacc';
elseif acc>=unacc && acc>=good && acc>=vgood
resultClass{testRow, rindex} = 'acc';
elseif good>=unacc && good>=acc && good>=vgood
resultClass{testRow, rindex} = 'good';
else
resultClass{testRow, rindex} = 'vgood';
end
k=k+1;
rindex = rindex+1;
end
end

for rindex=1:12
disp(getAccuracy(resultClass(:,rindex),new_label)*100);
end
end


knn(train_data,train_label,train_data,train_label,1);
disp('test --');
knn(train_data,train_label,new_data,new_label,0);

end



% Creates Dummy variable and concatenates it to the table
function table = createDummyConcat(data,s,l)
for i = s:l
data = horzcat(data,num2cell(dummyvar(categorical(data(:,i)))));
end
table = data;
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



