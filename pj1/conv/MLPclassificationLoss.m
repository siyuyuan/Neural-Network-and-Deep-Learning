function [f,g] = MLPclassificationLoss(w,X,y,nHidden,nLabels)
[nInstances,~] = size(X);
convWeights=reshape(w(1:25),5,5);
% Form Weights
inputWeights = reshape(w(25+1: 144*nHidden(1)+25),144,nHidden(1));
offset = 144*nHidden(1)+25;
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

gConv = zeros(size(convWeights)); 
f = 0;
if nargout > 1
    gInput = zeros(size(inputWeights));
    gOutput = zeros(size(outputWeights));
end

lambda = 0.001;
% Compute Output
for i = 1:nInstances
    convInput = reshape(X(i,2:257),16,16);
    convOutput = conv2(convInput,convWeights,'valid');
    Z = reshape(convOutput,1,144);
    ip = Z * inputWeights;
    fp = tanh(ip);
    z = fp * outputWeights;
    yhat = exp(z) ./ sum(exp(z));
    
    relativeErr = -log(yhat(y(i,:) == 1));
    f = f + relativeErr;
    
    if nargout > 1
        err = yhat - (y(i,:) == 1);
        gOutput = gOutput + fp' * err + lambda * outputWeights;
        clear backprop
        backprop = err * (repmat(sech(ip),nLabels,1).^2.*outputWeights');
        gInput = gInput + Z'* backprop + lambda*inputWeights;
        add_bias=[];
        add_bias= inputWeights;
        [ap,col]=size(add_bias);
        add_bias(:,col)=0;
        gInput=gInput+lambda*add_bias;
        backprop = backprop * inputWeights';
        reverseX = reshape(X(i,end:-1:2),16,16);
        backprop = reshape(backprop,12,12);
        gConv = gConv + conv2(reverseX, backprop, 'valid') + lambda*convWeights;

    end
    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:25)=gConv(:);
    g(26:144*nHidden(1)+25) = gInput(:);
    offset = 144*nHidden(1)+25;
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end
