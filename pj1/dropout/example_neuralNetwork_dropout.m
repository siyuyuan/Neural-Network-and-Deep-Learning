load digits.mat
tic;
[X, y] = transformations(X,y);
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
nHidden = [30];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
rand('seed',1);
w = unifrnd(-0.25,0.25,nParams,1);

% Train with stochastic gradient
maxIter = 300000;
stepSize = 1e-2;
funObj = @(w,i)MLPclassificationLoss_dropout(w,X(max(1,i-1):i,:),yExpanded(max(1,i-1):i,:),nHidden,nLabels);
iteration = [];
error = [];
key = 1;
w_opitmal = unifrnd(-0.5,0.5,nParams,1);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredict_dropout(w,Xvalid,nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
        
        if key > sum(yhat~=yvalid)/t
            key = sum(yhat~=yvalid)/t;
            w_optimal = w;
        end
        
        iteration = [iteration, iter-1];
        error = [error,sum(yhat~=yvalid)/t];
    end
    
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w = w - stepSize*g;
end

% Evaluate test error
yhat = MLPclassificationPredict_dropout(w_optimal,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
plot(iteration,error);
xlabel('Training iteration')  %x轴坐标描述
ylabel('validation error') %y轴坐标描述
toc
disp(['运行时间为：',num2str(toc)]);