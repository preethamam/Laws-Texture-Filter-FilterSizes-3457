function [featureVector, lawsTensorOutput, energyOutput, segMap]  = lawsFilter(image, filterType, ...
                            normtype, illumWindowSize, energyWindowSize, kClusters)
    % LAWSFILTER Applies Law's texture energy measures to an image.
    %
    %   [featureVector, lawsTensorOutput, energyOutput] = LAWSFILTER(image, filterType, ...
    %       normtype, illumWindowSize, energyWindowSize) applies Law's texture energy
    %       measures to the input image and returns the feature vector, tensor output,
    %       and energy output.
    %
    %   Inputs:
    %       image - Input image (grayscale or RGB).
    %       filterType - Type of Law's filter to use (3, 4, 5, or 7).
    %       normtype - Type of norm to use for feature vector ('L1', 'L2', 'infinity', 'frobenius').
    %       illumWindowSize - Size of the illumination window.
    %       energyWindowSize - Size of the energy window.
    %       kClusters -  K-means clusters number
    %
    %   Outputs:
    %       featureVector - Feature vector containing texture energy measures.
    %       lawsTensorOutput - Tensor output of Law's filters.
    %       energyOutput - Energy output of Law's filters.
    %       segMap - Segmentation map
    %
    %   Example:
    %       img = imread('example.jpg');
    %       [fv, lt, eo] = lawsFilter(img, 5, 'L2', 3, 5);
    %
    %   See also IMFILTER, RGB2GRAY, NORM.

    % Input args check
    if nargin < 5
        energyWindowSize = 5;
    end
    
    if nargin < 3
        normtype = 'L2';
        illumWindowSize = 3;
        energyWindowSize = 5;
    end
    
    if nargin < 2
        filterType = 4;
        normtype = 'L2';
        illumWindowSize = 3;
        energyWindowSize = 5;
    end
    
    % Laws multi-channels
    switch filterType
        case 3
            filters{1} = [ 1 2 1 ];           % level
            filters{2} = [-1 0 1];            % edge
            filters{3} = [-1 2 -1];           % spot
        case 4
            filters{1} = [ 1 4 6 4 1 ];       % level
            filters{2} = [-1 -2 0 2 1];       % edge
            filters{3} = [-1 0 2 0 -1];       % spot
            filters{4} = [1 -4 6 -4 1];       % ripple
        case 5
            filters{1} = [ 1 4 6 4 1 ];       % level
            filters{2} = [-1 -2 0 2 1];       % edge
            filters{3} = [-1 0 2 0 -1];       % spot
            filters{4} = [-1 2 0 -2 1];       % wave
            filters{5} = [1 -4 6 -4 1];       % ripple
        case 7
            filters{1} = [1 6 15 20 15 6 1];       % level
            filters{2} = [-1 -4 -5 0 5 4 1];       % edge
            filters{3} = [-1 -2 1 4 1 -2 -1];      % spot
            filters{4} = [-1 0 3 0 -3 0 1];        % wave
            filters{5} = [1 -2 -1 4 -1 -2 -1];     % ripple
            filters{6} = [1 -4 5 0 -5 4 -1];       % undulation
            filters{7} = [-1 6 -15 20 -15 6 -1];   % oscillation
        otherwise
            error('Require valid filter type.')
    end
    
    % Check for the RGB
    [height, width, channels] = size(image);
    if channels == 3
        image =  double(rgb2gray(image));
    end
    
    % Energy kernel
    energy_h = ones(energyWindowSize, energyWindowSize);
    
    % Prepare illumination filter kernel
    illumKernel = ones(illumWindowSize, illumWindowSize);
    centerPixel = ceil(illumWindowSize/2);
    illumKernel(centerPixel, centerPixel) = 0; 
    illumKernel = illumKernel / sum(illumKernel(:));
    
    % Remove effects of illumination
    imageFiltered = imfilter(image, illumKernel, 'conv', 'replicate');
    image = image - imageFiltered; % Subtract local mean.
    
    % Output stacks
    tensorOutput = zeros(height, width, size(filters,2) * size(filters,2));    
    indices = zeros(size(filters,2) * size(filters,2),2);
    
    counter = 1;
    % Filter responses
    for i = 1:size(filters,2)
        for j = 1:size(filters,2)
    
            % Form 3x3 or 5x5 or 7x7 tensors
            tensor     = filters{i}' * filters{j};
    
            % Create tensor filtered image
            filtered2D = imfilter(image, tensor, 'conv', 'replicate');
            tensorOutput(:,:,counter) = filtered2D;
            
            % Store indices
            indices(counter,:) = [i, j];
    
            % Counter to index vector
            counter = counter + 1;          
        end
    end
    
    % Get symmetric indices to combine symmetric pairs
    symmetricIndices = getSymmetricIndices(indices);
    
    % Feature vector initialization
    featureVector = zeros(length(symmetricIndices), 1);
    
    % Law's texture and energy combined output
    lawsTensorOutput = zeros(height, width, length(symmetricIndices));
    energyOutput = zeros(height, width, length(symmetricIndices));
    
    % Combine and obtain final maps
    counter = 1;
    for i = 1:length(symmetricIndices)
        
        % Get symmetric indices 
        symIdxs = symmetricIndices{i};
    
        % Case 1: If both filters are same
        if isscalar(symIdxs)
           lawsOut = tensorOutput(:,:,symIdxs);
        else % Case 2: If the pair equals its symmetric pair (Example: L5E5/E5L5)    
           lawsOut = mean(cat(3, tensorOutput(:,:,symIdxs(1)), tensorOutput(:,:,symIdxs(2))),3);
        end      
        
        % Law's combined output
        lawsTensorOutput(:,:,counter) = lawsOut;

        % Find the energy of the tensor filtered image
        energy2D   = imfilter(abs(lawsOut), energy_h, 'conv', 'replicate');
        energyOutput(:,:,counter) = energy2D;
        
        % Find a unique scalar for energy matrix (such as norm or others)
        switch normtype
            case 'L1'
                featureVector (counter) = norm(energy2D , 1);          % L1 norm
            case 'L2'
                featureVector (counter) = norm(energy2D , 2);          % L2 norm   
            case 'infinity'                
                featureVector (counter) = norm(energy2D , 'Inf');      % infinity norm
            case 'frobenius'
                featureVector (counter) = norm(energy2D , 'fro');      % Frobenius norm
        end   
    
        % Counter to index vector
        counter = counter + 1;      
    end    

    % Normalize the feature vector
    featureVector = featureVector /norm(featureVector);

    % Segmentation map and clustering
    segMaptic = tic;
    [N,M,L] = size(energyOutput);
    energyFeatureVec = reshape(energyOutput,[N*M, L]);
    
    % Get clusters indices/classes
    idx = kmeans(energyFeatureVec, kClusters);
    
    % Reshape to segmentation map
    segMap = reshape(idx, [N,M]);

    segMapruntime = toc(segMaptic);
    fprintf('Segmentation map execution time: %.4f seconds\n', segMapruntime)
end


function symmetricIndices = getSymmetricIndices(indices)
    % GETSYMMETRICINDICES Finds symmetric pairs of indices.
    %
    %   symmetricIndices = GETSYMMETRICINDICES(indices) returns the symmetric
    %   pairs of indices from the input indices.
    %
    %   Inputs:
    %       indices - Input indices.
    %
    %   Outputs:
    %       symmetricIndices - Symmetric pairs of indices.
    %
    %   Example:
    %       idx = [1 2; 2 1; 3 4; 4 3];
    %       symIdx = getSymmetricIndices(idx);
    %
    %   See also ISMEMBER, FLIP, FIND.

    % Initialize an array to keep track of processed pairs
    processed = false(size(indices, 1), 1);
    
    % Loop through each pair sequentially
    for i = 1:size(indices, 1)
        if processed(i) % Skip already processed indices
            continue;
        end
        
        pair = indices(i, :);                % Current pair
        symmetric_pair = flip(pair);         % Symmetric pair
        
        % Find all indices of this pair and its symmetric counterpart
        symmetric_indices = find(ismember(indices, [pair; symmetric_pair], 'rows'));
        
        % Mark these indices as processed
        processed(symmetric_indices) = true; 
        
        symIndxs{i} = symmetric_indices';
    end

    % Remove empty cells
    symmetricIndices = symIndxs(~cellfun(@isempty, symIndxs));    
end