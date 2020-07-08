function [model] = leastSquares(X,y)
% Solve least squares problem (assumes X'*X is invertible)
w = (X'*X)\X'*y;
model.w = w;
model.predict = @predict;
end