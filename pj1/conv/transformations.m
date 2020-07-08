function [trans_X, trans_y] = transformations(X,y)
[row,col] = size(X);
[r,c] = size(y);
A = randn(row,col);
a = randn(r,c);
flag = 0;
for k = 1:row
    p = unifrnd (0,1); 
    
    %translate
    if p <= 0.005
        flag = flag + 1;
        se = strel('line',5,90);
        temp = reshape(X(k,:), 16, 16);
        temp = imdilate(temp,se);
        A(flag,:) = reshape(temp, 1, 256);
        a(flag,:) = y(k,:);
    end
    
    %rotation 5
    if p <=0.01 && p> 0.005
        flag = flag+1;
        temp = reshape(X(k,:), 16, 16);
        temp = imrotate(temp,5,'crop');
        A(flag,:) = reshape(temp, 1, 256);
        a(flag,:) = y(k,:);
    end
    

    if p <=0.995 && p> 0.99
        flag = flag+1;
        temp = reshape(X(k,:), 16, 16);
        temp = imrotate(temp,-5,'crop');
        A(flag,:) = reshape(temp, 1, 256);
        a(flag,:) = y(k,:);
    end
    
    %flip
    if p >= 0.995
        flag = flag + 1;
        temp = reshape(X(k,:), 16, 16);
        temp = fliplr(temp);
        A(flag,:) = reshape(temp, 1, 256);
        a(flag,:) = y(k,:);
    end
end

trans_X = [X;A(1:flag,:)];
trans_y = [y;a(1:flag,:)];
end