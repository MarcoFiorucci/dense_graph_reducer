%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% noisyblockadjmat.m
% Generate a noisy adjacency matrix with noisy cluster over the diagonal. The computed matrix will have
% size = n_cluster * cluster_size
%
% input:
%    n_clusters: number of cluster
%    cluster_size: size of a single cluster 
%    internoise_lvl: percentage of noise between the clusters (0.0 for no noise)
%    intranoise_lvl: percentage of noise within a cluster (0.0 for completely connected clusters)
%    modality: the nature of the noise. Currently the supported values are 'weighted' and 'constant'
%    noise_val: the constant value to represent noise, used in combination with mode='constant'
%
% output:
%    mat: the noisy block adjacency matrix
%
%    the noise modalities are:
%    'weighted': replace the weight of an edge with a randomly generated number from 0.0 to 1.0
%    'constant': replace the weight of an edge with a constant value specified by noise_val parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function noisy_mat = noisyblockadjmat(n_clusters, cluster_size, varargin)

numvarargs = length(varargin);
if numvarargs > 4
    error('noisyblockadjmat:TooManyInputs', ...
        'requires at most 4 optional inputs');
end
    optargs = {0.0 0.0 'weighted' 1.0};
    optargs(1:numvarargs) = varargin;

    [internoise_lvl, intranoise_lvl, modality, noise_val] = optargs{:};
    mat_size = cluster_size * n_clusters;

    % interclass noise
    if internoise_lvl ~= 0.0
        mat = tril(rand(mat_size) < internoise_lvl, -1);

        if strcmp(modality, 'weighted')
            mat = mat .* rand(mat_size);
        elseif strcmp(modality, 'constant')
            mat = mat .* noise_val;
        else
            error('incorrect modality');
        end
    else
        mat = zeros(mat_size);
    end

    for i=1:n_clusters
        % intraclass noise
        cl = tril(ones(cluster_size), -1);
        if intranoise_lvl ~= 0.0
            noise = tril(rand(cluster_size) < intranoise_lvl, -1);
            if strcmp(modality, 'weighted')
                noise = noise .* rand(cluster_size);
            elseif strcmp(modality, 'constant')
                noise = noise .* (1.0 - noise_val);
            else
                error('incorrect modality');
            end
            cl = cl - noise;
        end
        mat(((i - 1) * cluster_size + 1):(i * cluster_size), ((i - 1) * cluster_size + 1):(i * cluster_size)) = cl;
    end
    noisy_mat = mat + mat';
end