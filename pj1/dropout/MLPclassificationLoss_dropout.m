function [f,g] = MLPclassificationLoss_dropout(w,X,y,nHidden,nLabels)

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

f = 0;
if nargout > 1
    gInput = zeros(size(inputWeights));
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
    gOutput = zeros(size(outputWeights));
end

lambda = 0.001;
% Compute Output
for i = 1:nInstances
    
    dropout_matrix{1}=rand(size(X( i ,:)))> dropout_prob;
    X(i,:)= X(i,:).*dropout_matrix{1};
    
    ip{1} = X(i,:)*inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        
        dropout_matrix{h}=rand(size(fp{h-1})) > dropout_prob;
        fp{h-1}= fp{h-1}.*dropout_matrix{h};
        
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    z = fp{end}*outputWeights;
    yhat = exp(z) ./ sum(exp(z));
    
    relativeErr = -log(yhat(y(i,:) == 1));
    f = f + relativeErr ;
    
    if nargout > 1
        err = yhat - (y(i,:) == 1);

        % Output Weights
        %for c = 1:nLabels
        %    gOutput(:,c) = gOutput(:,c) + err(c)*fp{end}';
        %end
        gOutput = fp{end}' * err + lambda * outputWeights;

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
            %for c = 1:nLabels
            %    backprop(c,:) = err(c)*(sech(ip{end}).^2.*outputWeights(:,c)');
            %    gHidden{end} = gHidden{end} + fp{end-1}'*backprop(c,:);
            %end
            backprop = err*(repmat(sech(ip{end}).^2,nLabels,1).*outputWeights');
            fp{end-1}=fp{end-1}.* dropout_matrix{end};
            gHidden{end} = gHidden{end} + fp{end-1}'*backprop+ lambda*hiddenWeights{length(nHidden)-1};
            backprop = sum(backprop,1);
            add_bias=[];
            add_bias= hiddenWeights{length(nHidden)-1};
            [~,col]=size(add_bias);
            add_bias(:,col)=0;
            gHidden{end}=gHidden{end}+lambda*add_bias;

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
                fp{h}=ip{h}.*dropout_matrix{h+1};
                gHidden{h} = gHidden{h} + fp{h}'*backprop+ lambda * hiddenWeights{h};
                add_bias=[];
                add_bias= hiddenWeights{h};
                [~,col]=size(add_bias);
                add_bias(:,col)=0;
                gHidden{h}=gHidden{h}+lambda*add_bias;
            end

            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            X(i,:)=X(i,:).*dropout_matrix{1};
            gInput = gInput + X(i,:)'*backprop+ lambda*inputWeights;
            add_bias=[];
            add_bias= inputWeights;
            [~,col]=size(add_bias);
            add_bias(:,col)=0;
            gInput=gInput+lambda*add_bias;
        else
           % Input Weights
            %for c = 1:nLabels
            %    gInput = gInput + err(c)*X(i,:)'*(sech(ip{end}).^2.*outputWeights(:,c)');
            %end
            backprop = err*(repmat(sech(ip{end}),nLabels,1).^2.*outputWeights');
            X(i,:)=X(i,:).*dropout_matrix{1};
            gInput = gInput + X(i,:)'* backprop + lambda*inputWeights;
            add_bias=[];
            add_bias= inputWeights;
            [~,col]=size(add_bias);
            add_bias(:,col)=0;
            gInput=gInput+lambda*add_bias;
        end

    end
    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end
