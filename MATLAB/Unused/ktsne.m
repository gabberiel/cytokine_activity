function [Ytr, Ytest, traininds, testinds] = ktsne(X, nTrain, perpl, plotresult, k, Ytr, no_dims)
% Implementation of kernel t-SNE from 
% http://www.sciencedirect.com/science/article/pii/S0925231214007036
% X      - N x F input data array
% nTrain - number of training point to use out of the N possible F-dimensional data 
%          points
% perpl  - perplexity parameter for t-SNE (30 is the default for compute_mapping)
% plotresult - logical for plotting the result
% k      - factor that multiplies sigma2
% Ytr    - (optional) t-SNE output for the training set
%          If not specified it will be computed in O(N*log(N)) time with N^2 memory
% Ytest  - kernel t-SNE result from all data points not in the training set

% select training and test set
if isscalar(nTrain)
    skip = floor(size(X, 1) / nTrain);
    traininds = 1:skip:nTrain * skip;    
else
    traininds = nTrain;
end
testinds = setdiff(1:size(X, 1), traininds);
Xtr = X(traininds, :);
Xtest = X(testinds, :);

% t-SNE
if ~exist('Ytr', 'var') || isempty(Ytr)   
    % no_dims = 2;  % Could potentially be 3    
    labels = 10*log10(sum(Xtr.^2, 2));
    maxiter = 200;
    videofile = '';  % sprintf('b1_s4_perp%d_tsne.avi', perpl);
    tic;
    [Ytr, ~] = compute_mapping(Xtr, 'tSNE', no_dims, size(Xtr, 2), perpl, labels, maxiter, videofile);
    toc;
    % save the result in case the function crashes (out of memory, etc.)
    save('temp.mat', 'Ytr', 'Xtr', 'Xtest');
end

if ~exist('k', 'var') || isempty(k)
    k = 1;
end

%{
[~, systemview] = memory();
svpma = systemview.PhysicalMemory.Available;
if isscalar(nTrain) && size(X, 1) > nTrain && 8 * (size(X, 1) - nTrain) * nTrain >= svpma
    Ytest = [];
    disp('ktsne would run out of memory, exiting early');
    return
end
%}
if isempty(testinds)
    Ytest = [];
    disp('no testing set, exiting early');
    return
end

% Pairwise distances
% compute_mapping will internally compute Dtr so don't fill up the memory
% until after it completes
Dtr = pdist2(Xtr, Xtr, 'squaredeuclidean');
Dtest = pdist2(Xtr, Xtest, 'squaredeuclidean');

% determine sigma
Dtest(Dtest == 0) = Inf;
mv = min(Dtest, [], 2);
sigma2 = k * sqrt(-0.5 * mv / log(eps));
Dtest(~isfinite(Dtest)) = 0;

% compute Ktr (rename Dtr to Ktr to save memory)
Ktr = Dtr;
clear Dtr
Ktr = exp(bsxfun(@rdivide, -0.5 * Ktr, sigma2));
Ktr = bsxfun(@rdivide, Ktr, sum(Ktr, 2));

% compute A
A = Ktr \ Ytr;

% compute Ktest (rename Dtest to Ktest to save memory)
Ktest = Dtest;
clear Dtest

Ktest = exp(bsxfun(@rdivide, -0.5 * Ktest, sigma2));
% when sum(Ktest, 1) underflows we will get Inf which will lead to NaNs

Ktest = bsxfun(@rdivide, Ktest, sum(Ktest, 1));

% compute Ytest
Ytest = Ktest' * A;

% Plot
if plotresult    
    cols = [ones(size(Ytest, 1), 1); 2*ones(size(Ytr, 1), 1)];
    x = [Ytest(:, 1); Ytr(:, 1)];
    y = [Ytest(:, 2); Ytr(:, 2)];
    figure;scatter(x, y, 5, cols, 'fill');
end