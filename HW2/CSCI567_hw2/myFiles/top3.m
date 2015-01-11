function [ dict, data, label ] = top3(folder)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

dict = importDict('vocab.dat', 0);
ddHam = dir(strcat('./hw2_data/spam/',folder,'/ham'));
fileNamesHam = {ddHam.name};
ddSpam = dir(strcat('./hw2_data/spam/',folder,'/spam'));
fileNamesSpam = {ddSpam.name};

data = zeros(length(fileNamesHam)+length(fileNamesSpam)-4,length(dict));
label = zeros(length(fileNamesHam)+length(fileNamesSpam)-4,1);

% Ham
for i = 3:length(fileNamesHam)
    fdata = fileread(fileNamesHam{i});
    vals = textscan(fdata,'%s','delimiter',' .,?');
    vals = vals{1};
    
    for j = 1:length(vals)
        index = getIndexInDict(dict,vals{j});
        if isempty(index)~=1
            data(i-2,index) = data(i-2,index)+1;
        end
    end
    label(i-2) = 1;
    
end


%spam
for i = 3:length(fileNamesSpam)
    fdata = fileread(fileNamesSpam{i});
    vals = textscan(fdata,'%s','delimiter',' .,?');
    vals = vals{1};
    
    for j = 1:length(vals)
        index = getIndexInDict(dict,vals{j});
        if isempty(index)~=1
            data(length(fileNamesHam)-2+i-2,index) = data(length(fileNamesHam)-2+i-2,index)+1;
        end
    end
    label(length(fileNamesHam)-2+i-2) = 0;
end


if strcmp(folder,'train')
    sumData = sum(data)';
    [V,I] = sort(sumData,'descend');
    disp('Top 3 Words -- ');
    fprintf('{ ');
    for i = 1:3
        fprintf('(%s,%d) ',dict{I(i)},sumData(I(i)));
    end
    fprintf('}\n'); 
end

end

function index = getIndexInDict(dict, refString)
    index = find(strcmp(dict,refString));
end
