function [ class ] = improved_kmeans( training_data, k )

% means = zeros(k,size(training_data,2));

means = datasample(training_data,k,'Replace',false);
class = zeros(size(training_data,1),1);

for it = 1:10
    sum = zeros(size(training_data,1),size(means,1));
    for j = 1:size(training_data,2)
        sum = sum + pow2((bsxfun(@minus,training_data(:,j),means(:,j)')));
    end
    dist_matrix = sqrt(sum);
    [Y,new_class] = min(dist_matrix,[],2);
    
    if all(new_class == class)
        break;
    end
    
    class = new_class;
    for j = 1:k
       means(j,:) = mean(training_data(new_class==j,:));
    end
end


% % while 1
% for it = 1:10
%     for j = 1:k
%         dist =  pdist2(training_data,repmat(means(j,:),size(training_data,1),1),'euclidean');
%         dist_matrix(:,j) = dist(:,1);
%     end
%     
%     [Y,new_class] = min(dist_matrix,[],2);
%     
%     if all(new_class == class)
%         break;
%     end
%     
%     class = new_class;
%     for j = 1:k
%        means(j,:) = mean(training_data(new_class==j,:));
%     end
%     
% end
end

function [D] = myDistance(X, Y)
D = bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);
end