function [w,b] = trainsvm(train_data, train_label, C)
% Train linear SVM (primal form)
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  C: tradeoff parameter (on slack variable side)
%
% Output:
%  w: feature vector (column vector)
%  b: bias term
%
% CSCI 576 2014 Fall, Homework 3


% Slacked based SVM
N = size(train_data,1);
D = size(train_data,2);

H = eye(D+1+N); % w + b + ksi
H(D+1:end,D+1:end) = 0; %quadratic is only for w

f = C * ones(D+1+N,1); % this condition is only for ksi
f(1:D+1) = 0; % so making others 0 -> w+b

A = repmat(train_label,1,D+1) .* [train_data ones(N,1)];  % for y.x & y.b
A = -1 * [A eye(N)]; % adding eye(N) for ksi
b = -1 * ones(N, 1);

%lower bound
lb = [-inf(D+1, 1);zeros(N, 1)];

opts = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');
[result, fval]= quadprog(H, f, A, b, [], [], lb, [],[],opts);

w = result(1:D);
b = result(D+1);


% Dual Version

% 
% %Defining H
% H = (train_data*train_data').*(train_label*train_label');
% 
% f = -1*ones(size(train_data,1),1);
% 
% Aeq = [[train_label'];[zeros(size(train_label,1)-1,size(train_label,1))]];
% beq = zeros(size(train_label,1),1);
% 
% 
% A = -1*eye(size(train_data,1));
% %Upper lower bounds for alpha
% lb = zeros(size(train_data,1),1);
% ub = C*ones(size(train_data,1),1);
% 
% 
% x = quadprog(H,f,A,lb,Aeq,beq,lb,ub);

