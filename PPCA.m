%Junyu Chen

clear all; close all;
%% load data
train_x = loadMNISTImages('data/train-images-idx3-ubyte');
train_y = loadMNISTLabels('data/train-labels-idx1-ubyte');
test_x = loadMNISTImages('data/t10k-images-idx3-ubyte');
test_y = loadMNISTLabels('data/t10k-labels-idx1-ubyte');

%% centering data
mu = mean(train_x,2);
train_x_centered = bsxfun(@minus,train_x,mu);
test_x_centered = test_x - mean(test_x,2);

%% initialize W
% Begin Probabilistic PCA
[d, n] = size(train_x_centered);
dimension = [50 100];
correctness = [];
for dem_i = 1:2
    k = dimension(dem_i);
    I = eye(k);
    S = sum(sum((train_x_centered.^2))); % total norm of X
    W = randn(d,k); % Initialize W
    sigmaSqure = 1/randg; % Initialize sigma
    llp = [];
for iterNum = 1:20
%% calculating log likelihood
    % calculating M
    M = W'*W+sigmaSqure*I;
    % calculating U
    U = chol(M);
    % calculating ln(M)
    lnM = 2*sum(log(diag(U))) + (d-k)*log(sigmaSqure);
    % calculating wx
    wx = W'*train_x_centered;
    % calculating T
    T = U'\ wx;
    % calculating tr((M^-1)S)
    tmpT = sum(sum(T.^2));
    trinvMS= (S - tmpT)/(sigmaSqure*n);
    % log liklihood
    llp =[llp -n/2*(d*log(2*pi)+lnM+trinvMS)];
%% Using E step to obtain mean and variance of hidden variables z_n
    % inv(M) x W' x X
    Ezn = M\wx;
    % get V
    V = inv(U);
    % inv(M) = V*V'
    invM = V*V';
    % Get E[ZnZn']
    Eznzn = n*sigmaSqure*(invM)+Ezn*Ezn'; 
%% 	Using M-step, obtain new values for W and sigma and update old parameters with these new parameters    
    Ezntmp = chol(Eznzn);
    Wtmp = train_x_centered*Ezn';
    Wnew = (Wtmp/Ezntmp)/Ezntmp';         
    WU = Wnew*Ezntmp';
    signmaTmp1 = dot(Ezn(:),wx(:));
    signmaTmp2 = dot(WU(:),WU(:));
    sigmaSqurenew = (S-2*signmaTmp1+signmaTmp2);
    
    % update w and sigmaSqure
    sigmaSqure = sigmaSqurenew/(n*d);
    W = Wnew;
end

% Begin Probabilistic PCA

%% gaussian classifier
Mtmep = W'*W + sigmaSqure*I;

% get the training set and test set projections
train_prj = Mtmep\W'*train_x_centered;
test_prj = Mtmep\W'*test_x_centered;
train_prj = train_prj';
test_prj = test_prj';

%% find mean & covariance for each class
class_mean = zeros(10,k);
class_cov = {};
num_of_samples = [];
for i = 0:9
    num_of_samples = [num_of_samples length(find(train_y == i))];
    oneClass = train_prj(find(train_y == i),:);
    class_mean(i+1,:) = mean(oneClass,1);
    class_cov = [class_cov cov(oneClass)]; 
end


%% individual cov
totalNumSamples = sum(num_of_samples);
correct = 0;
testNum = 10000;
for test_index = 1:testNum
P_i = [];   
for class_index = 1:10
    M = (test_prj(test_index,:)-class_mean(class_index,:))';
    Pc_i = num_of_samples(class_index)/totalNumSamples;
    %Pc_i = 1/10;
    tmp = log(1)-log((2*pi)^(k/2)*det(class_cov{class_index}));
    P_i = [P_i log(Pc_i)+tmp+(-1/2*M'/(class_cov{1,class_index})*M)];
end
[val, index] = max(P_i);
if((index-1) == test_y(test_index))
  correct = correct + 1;
end
end
figure;
plot(llp)
%%
correctness = [correctness correct/testNum];
end
%%
y = [correctness(1) correctness(2)];
figure;
bar(y)
set(gca,'xticklabel',{'k = 50','k = 100'})
title('Results for different k values')