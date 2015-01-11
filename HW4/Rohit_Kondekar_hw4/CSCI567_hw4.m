function [  ] = Untitled( input_args )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

path = pwd;
addpath(genpath(strcat(path,'/myFiles')));
addpath(genpath(strcat(path,'/hw4_data')));

data = importfile('2DGaussian.csv');
class = data(:,1);
training_data = data(:,2:end);


% %Part A
for k = [2,3,5]
    new_class = my_kmeans(training_data,k);
    figure
    scatter(training_data(:,1),training_data(:,2),50,new_class,'filled');
    title(strcat('KMeans clustering with K=',num2str(k))); 
end

%PartB
markers = {'-.r*','--mo',':bs','-.','-x'};
cc=jet(12);
figure; 
hold on;
for i = 1:5
    distances = kmeansPartB(training_data,4);
    plot(distances,markers{mod(i,numel(markers))+1},'color',cc(i+2,:),'linewidth',2);
end
legend('Run 1','Run 2','Run 3','Run 4','Run 5');
title('Objective Function fore K Means in different runs over 50 Iterations'); 

%5.3
imageMatrix = imread('hw4.jpg');

for i = [3,8,15]
    vector = stackImage(imageMatrix);
    class = my_kmeans(vector,i);
    newVector = vectorQuatize(vector,class,i);
    newMatrix = unstackImage(newVector,imageMatrix);
    figure
    image(newMatrix);
    title(strcat('Image Compression with KMeans with K=',num2str(i))); 
end

end


function [newVector] = vectorQuatize(vector,class,k)

    
    for i=1:k
       if(sum(class==i)>0)
           m1 = mean(vector(class==i,1));
           m2 = mean(vector(class==i,2));
           m3 = mean(vector(class==i,3));
           vector(class==i,1) = m1;
           vector(class==i,2) = m2;
           vector(class==i,3) = m3;
       end
    end
    newVector = vector;
end


function [outputMatrix] = stackImage(matrix)

    outputMatrix = zeros(size(matrix,1)*size(matrix,2),3);
    ct = 1;
    for i = 1:size(matrix,1)
        for j = 1:size(matrix,2);
           for k = 1:3
               outputMatrix(ct,k) = matrix(i,j,k);
           end
           ct = ct+1;
        end
    end

end

function [orig_matrix] = unstackImage(matrix,orig_matrix)

    col = 0;
    row = 0;
    for i = 1:size(matrix,1)
        
        if col==0
            col=1;
            row=row+1;
        end
        
        for k = 1:3
            orig_matrix(row,col,k) = matrix(i,k);
        end
        
        col = mod((col+1),size(orig_matrix,2)+1);
    end
end




