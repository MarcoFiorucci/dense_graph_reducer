function simMat = img2simmat(imgPath)
    I = imread(imgPath);
    I = rgb2gray(I);
    I = im2double(I);
    I = imresize(I, 0.25);
    nodes = size(I, 1) .* size(I, 2);
    X = reshape(I, [nodes, 1]);
    D = pdist2(X, X, 'euclidean');
    S = exp(-(D / 2.^2));
    numVx = size(S, 1);
    S(1:numVx + 1:end) = 0;
    simMat = S;
end




