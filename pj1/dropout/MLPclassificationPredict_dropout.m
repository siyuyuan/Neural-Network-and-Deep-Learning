function [y] = MLPclassificationPredict_dropout(w,X,nHidden,nLabels)
[nInstances,nVars] = size(X);

dropout_prob = 0.5;

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

% Compute Output
for i = 1:nInstances
    
    dropout_matrix{1}=(rand(size(X( i ,:))) > dropout_prob);
    X(i,:)= X(i,:).*dropout_matrix{1};

    ip{1} = X(i,:)*inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        
        dropout_matrix{h}=(rand(size(fp{h-1})) > dropout_prob);
        fp{h-1}= fp{h-1}.*dropout_matrix{h};
        
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    y(i,:) = fp{end}*outputWeights;
end

for i = 1:nInstances
    y(i,:) = exp(y(i,:)) ./ sum(exp(y(i,:)));
end

[v,y] = max(y,[],2);
%y = binary2LinearInd(y);
