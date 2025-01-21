%% Start parameters
%--------------------------------------------------------------------------
clear; close all; clc;
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);
Start = tic;

%% Inputs
image = imread("images/texture_02.jpg");
filterType = 4;                     % Law's filter to use (3, 4, 5, or 7)
normtype = 'L2';                    % Matrix norm: 'L1', 'L2', 'Inf', 'fro'
illumWindowsize = 3;                % Illumination smoothing window size
energyWindowsize = 7;               % Energy window size

%% Process imagenusing the Law's texture filter
lawstic = tic;
[featureVector, tensorOutput, energyOutput]  = lawsFilter(image, filterType, normtype, illumWindowsize,...
                                                          energyWindowsize);
lawsruntime = toc(lawstic);
fprintf('Law''s texture filter execution time: %.4f seconds\n', lawsruntime)

%% Montage plot
figure;
montPlot = montage(uint8(tensorOutput));
exportgraphics(gca, 'assets/lawsplot.png')

figure;
montPlot2 = montage(mat2gray(energyOutput));
exportgraphics(gca, 'assets/energyplot.png')

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




