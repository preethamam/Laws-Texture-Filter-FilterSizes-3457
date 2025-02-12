%% Start parameters
%--------------------------------------------------------------------------
clear; close all; clc;
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);
Start = tic;

%% Inputs
image = imread("images/comb.jpg");
filterType = 4;                     % Law's filter to use (3, 4, 5, or 7)
normtype = 'L2';                    % Matrix norm: 'L1', 'L2', 'Inf', 'fro'
illumWindowsize = 3;                % Illumination smoothing window size
energyWindowsize = 7;               % Energy window size
kClusters = 5;                     % K-means clusters number

%% Process imagenusing the Law's texture filter
lawstic = tic;
[featureVector, tensorOutput, energyOutput, segMap]  = lawsFilter(image, filterType, normtype, illumWindowsize,...
                                                          energyWindowsize, kClusters);
lawsruntime = toc(lawstic);
fprintf('Law''s texture filter execution time: %.4f seconds\n', lawsruntime)

%% Montage plot
figure;
montPlot = montage(uint8(tensorOutput));
exportgraphics(gca, 'assets/lawsplot.png')

figure;
montPlot2 = montage(mat2gray(energyOutput));
exportgraphics(gca, 'assets/energyplot.png')

figure;
t = tiledlayout(1,2,TileSpacing="tight",Padding="compact");
nexttile; imshow(image); title('Original')
nexttile; imshow(label2rgb(segMap)); title('Segmentation Map')
exportgraphics(gcf, 'assets/segmap.png')

%% End parameters
%--------------------------------------------------------------------------
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);
statusFclose = fclose('all');
if(statusFclose == 0)
    disp('All files are closed.')
end
Runtime = toc(Start);
disp(Runtime);




