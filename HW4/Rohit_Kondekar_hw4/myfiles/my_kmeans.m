function [ class ] = my_kmeans( training_data, k )

cluster_means = datasample(training_data,k,'Replace',false);
class = zeros(size(training_data,1),1);

dist_matrix = zeros(size(training_data,1),k);

while 1
    dist_matrix = pdist2(training_data,cluster_means);
%     for j = 1:k
%         dist =  pdist2(training_data,repmat(means(j,:),size(training_data,1),1),'euclidean');
%         dist_matrix(:,j) = dist(:,1);
%     end
%     
    
    [Y,new_class] = min(dist_matrix,[],2);
  
    if all(new_class == class)
        break;
    end
    
    class = new_class;
    for j = 1:k
       cluster_means(j,:) = mean(training_data(new_class==j,:));
    end

end
end