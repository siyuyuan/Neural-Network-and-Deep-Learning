%改过第十个神经网络的。
function [y] = MLPclassificationPredict(w,X,nHidden,nLabels)
%先把放入的5000*256变成6*6的矩阵。
[nInstances,~] = size(X);
%conv weights
convWeights=reshape(w(1:25),5,5);
% Form Weights
inputWeights = reshape(w(25+1: 144*nHidden(1)+25),144,nHidden(1));
offset = 144*nHidden(1)+25;
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

% Compute Output
for i = 1:nInstances
    convInput = reshape(X(i,2:257),16,16);
    convOutput = conv2(convInput,convWeights,'valid');
    Z = reshape(convOutput,1,144);
    ip = Z * inputWeights;
    fp = tanh(ip);
    y(i,:) = fp * outputWeights;
end

for i = 1:nInstances
    y(i,:) = exp(y(i,:)) ./ sum(exp(y(i,:)));
end

[v,y] = max(y,[],2);
%y = binary2LinearInd(y);
