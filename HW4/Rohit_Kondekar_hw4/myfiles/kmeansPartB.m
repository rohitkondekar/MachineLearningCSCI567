function [ distances ] = kmeansPartB( training_data, k )

cluster_means = datasample(training_data,k,'Replace',false);
class = zeros(size(training_data,1),1);
dist_matrix = zeros(size(training_data,1),k);

distances = zeros(50,1);


for it = 1:50 %iterations
    
    dist_matrix =  pdist2(training_data,cluster_means);
%     for j = 1:k
%         dist =  pdist2(training_data,repmat(means(j,:),size(training_data,1),1),'euclidean');
%         dist_matrix(:,j) = dist(:,1);
%     end
    
    [Y,new_class] = min(dist_matrix,[],2);
    
%     if all(new_class == class)
%         break;
%     end
    
    class = new_class;
    for j = 1:k
       cluster_means(j,:) = mean(training_data(new_class==j,:));
    end
    
    val = 0;
    for j = 1:k
         t = sum(pdist2(training_data(new_class==j,:),repmat(cluster_means(j,:),size(training_data(new_class==j,:),1),1),'euclidean'));
         val = val + t(1,1);
    end
    distances(it) = val;
end

end

function [val] = myDistance(M)
    val = sqrt(sum(M.^2,2));
end

% 
% distances = zeros(50,1);
% for it = 1:50 %iterations
%     
%    for j = 1:k
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
%     dist = 0;
%     for j = 1:k
%         t = sum(pdist2(training_data(new_class==k,:),repmat(means(j,:),size(training_data,1),1),'euclidean'));
%         dist = dist + t(1,1);
%     end
%     distances(it) = dist;
% end
% end



