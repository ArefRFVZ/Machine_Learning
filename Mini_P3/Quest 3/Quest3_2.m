%----------------------------------------------------------------------
% anfis_genfis2_multioutput.m
%
% This script implements a multi-output ANFIS modeling approach for a
% steam generator dataset with 4 inputs and 4 outputs. Each output is
% modeled by its own ANFIS (single-output) structure, trained using
% subtractive clustering via genfis2 and then refined by anfis.
%
% The dataset is assumed to have 9 columns:
%   1) time (unused for modeling)
%   2) u1: Fuel            (scaled 0-1)
%   3) u2: Air             (scaled 0-1)
%   4) u3: Reference level (inches)
%   5) u4: Disturbance     (load level)
%   6) y1: Drum pressure (PSI)
%   7) y2: Excess Oxygen (%)
%   8) y3: Water level in drum
%   9) y4: Steam Flow (Kg/s)
%
% Author:      [Your Name]
% Institution: [Your Institution]
% Date:        [Date]
% Contact:     [Your Email]
%----------------------------------------------------------------------


%% 0) INITIALIZATION & PARAMETERS
% -----------------------------------------------------------------
rng(73);               % For reproducibility is 13 becuse my student number is 40010273
clusterRadius = 0.6;   % Tuning parameter for genfis2 (subtractive clustering)
maxEpochs     = 50;   % # of training epochs
errorGoal     = 0;     % Desired training error (0 for no specific goal)
initStepSize  = 0.01;  % Initial step size
stepSizeDecr  = 0.99;   % Step size decrease factor
stepSizeIncr  = 1.01;   % Step size increase factor

% Options array for anfis
trainOptions = [maxEpochs errorGoal initStepSize stepSizeDecr stepSizeIncr];
% Display options: [DisplayInfo, DisplayError, DisplayStepSize, DisplayFinal]
dispOptions  = [1, 0, 0, 1];



% Optimization method:
%   0 -> Backpropagation
%   1 -> Hybrid (Backpropagation + Least-Squares)
optMethod = 1;


%% 1) LOAD AND PREPARE DATA
% -----------------------------------------------------------------
% Adjust the filename/path to your actual data file:
rawData = load('steamgen.dat');  %#ok<*LOAD> 
% rawData is assumed to be an N x 9 matrix:
%   col 1 -> time
%   col 2..5 -> inputs  (u1,u2,u3,u4)
%   col 6..9 -> outputs (y1,y2,y3,y4)

if size(rawData,2) < 9
    error('Expected at least 9 columns in data: time + 4 inputs + 4 outputs.');
end

% Separate inputs (4 columns) and outputs (4 columns)
inputs_raw  = rawData(:,2:5);
outputs_raw = rawData(:,6:9);

% (Optional) Display a quick diagnostic plot:

figure; plot(rawData(:,1), rawData(:,2));
title('First Input vs. Time'); xlabel('Time index'); ylabel('Fuel Input');


%% 2) DATA PREPROCESSING: NORMALIZATION
% -----------------------------------------------------------------
% Normalize both inputs and outputs to [0, 1] range (min-max).
[inputs_norm, inMin, inMax]   = normalizeMinMax(inputs_raw);
[outputs_norm, outMin, outMax] = normalizeMinMax(outputs_raw);

% Combine for splitting
dataNorm = [inputs_norm, outputs_norm];
N = size(dataNorm,1);

%% 3) TRAIN/VALIDATION/TEST SPLIT
% -----------------------------------------------------------------
trainRatio = 0.70;
valRatio   = 0.15;
testRatio  = 0.15;

% Check that these ratios sum up to 1.0 (or close)
if abs(trainRatio + valRatio + testRatio - 1.0) > 1e-9
    error('Train/Val/Test ratios must sum to 1.0.');
end

% Shuffle indices
randIdx = randperm(N);

nTrain = round(trainRatio*N);
nVal   = round(valRatio*N);
nTest  = N - nTrain - nVal;  % leftover

idxTrain = randIdx(1:nTrain);
idxVal   = randIdx(nTrain+1 : nTrain+nVal);
idxTest  = randIdx(nTrain+nVal+1 : end);

trainData = dataNorm(idxTrain,:);
valData   = dataNorm(idxVal,:);
testData  = dataNorm(idxTest,:);

% Separate into X (inputs) and Y (outputs)
X_train = trainData(:,1:4);
Y_train = trainData(:,5:8);

X_val   = valData(:,1:4);
Y_val   = valData(:,5:8);

X_test  = testData(:,1:4);
Y_test  = testData(:,5:8);


%% 4) BUILD AND TRAIN ANFIS MODELS (ONE PER OUTPUT)
% -----------------------------------------------------------------
% We will have 4 separate FIS structures, each modeling one output.
% For each output dimension, we do:
%
%   (a) Generate initial FIS using genfis2 (subtractive clustering)
%   (b) Train (fine-tune) the FIS using anfis in hybrid mode
%
% Store the final FIS in a cell array `fisList`.

fisList = cell(1,4);       % store final trained FIS for each output
trainErrorList = cell(1,4);
valErrorList   = cell(1,4);

for outIdx = 1:4
    
    % Prepare data for genfis2: [X, Y] with single-column Y
    trainDataSingle = [X_train, Y_train(:,outIdx)];
    
    % STEP (a): Build initial FIS using subtractive clustering
    % The clusterRadius parameter strongly affects the number of rules.
    initFIS = genfis2(X_train, Y_train(:,outIdx), clusterRadius);
    
    % STEP (b): Train the FIS using anfis
    [fisTrained, trainError, ~, fisFinal, valError] = ...
        anfis(trainDataSingle, initFIS, trainOptions, dispOptions, ...
              [X_val, Y_val(:,outIdx)], optMethod);
    
    % Store results
    fisList{outIdx}       = fisFinal;
    trainErrorList{outIdx} = trainError;
    valErrorList{outIdx}   = valError;
end


%% 5) MODEL EVALUATION (TRAIN / VAL / TEST)
% -----------------------------------------------------------------
% Evaluate performance for each output. We compute RMSE on 
% training, validation, and testing sets in normalized scale.

rmseTrain = zeros(1,4);
rmseVal   = zeros(1,4);
rmseTest  = zeros(1,4);

for outIdx = 1:4
    % Training predictions
    yhat_train = evalfis(fisList{outIdx}, X_train);
    rmseTrain(outIdx) = sqrt(mean((Y_train(:,outIdx) - yhat_train).^2));
    
    % Validation predictions
    yhat_val = evalfis(fisList{outIdx}, X_val);
    rmseVal(outIdx) = sqrt(mean((Y_val(:,outIdx) - yhat_val).^2));
    
    % Testing predictions
    yhat_test = evalfis(fisList{outIdx}, X_test);
    rmseTest(outIdx) = sqrt(mean((Y_test(:,outIdx) - yhat_test).^2));
end







% Display the results
disp('-------------------------------------------------------------');
disp('Normalized RMSE for each output:');
disp(table((1:4)', rmseTrain', rmseVal', rmseTest',...
     'VariableNames',{'Output','TrainRMSE','ValRMSE','TestRMSE'}));

%% (Optional) De-normalize RMSE
% You can de-normalize the errors if you want physical units. Example:
  yhat_test_de   = yhat_test*(outMax(outIdx)-outMin(outIdx)) + outMin(outIdx);
  Y_test_de      = Y_test(:,outIdx)*(outMax(outIdx)-outMin(outIdx)) + outMin(outIdx);
% Then compute the RMSE in original scale.  

%% 6) VISUALIZATION


% -----------------------------------------------------------------
% 6a) Plot training & validation error over epochs for each output
figure('Name','ANFIS Training and Validation Errors');
for outIdx = 1:4
    subplot(2,2,outIdx);
    plot(trainErrorList{outIdx},'LineWidth',1.5); hold on;
    plot(valErrorList{outIdx},  'LineWidth',1.5);
    title(['Output #',num2str(outIdx),' - Learning Curve']);
    xlabel('Epoch'); ylabel('RMSE');
    legend('TrainError','ValError'); grid on;
end



% 6b) Plot predictions vs. targets for Test set
figure('Name','Outputs vs Targets (Test Set)');
for outIdx = 1:4
    subplot(2,2,outIdx);
    yhat_test = evalfis(fisList{outIdx}, X_test);
    plot(Y_test(:,outIdx),'b','LineWidth',1); hold on;
    plot(yhat_test,'r','LineWidth',1);
    title(['Output #',num2str(outIdx),' on Test Data']);
    xlabel('Sample'); ylabel('Normalized Value');
    legend('Target','Predicted'); grid on;
end

% 6c) (Optional) Inspect membership functions for one of the FIS
figure;
for i=1:4
    for j=1:4
    subplot(4,4,j+((i-1)*4))
    plotmf(fisList{i}, 'input', j);
    title(['MFs of input ', num2str(j), ' for Output #', num2str(i), ' FIS']);
    end
end

% 6d) (Optional) Show fuzzy inference system structure

for i=1:4            
    figure; plotfis(fisList{i});title([' for Output #', num2str(i), ' FIS']);  % diagram of the fuzzy system
end

%--------------------------------------------------------------------------
%  HELPER FUNCTION: Min-Max Normalization
%--------------------------------------------------------------------------
function [dataNorm, dataMin, dataMax] = normalizeMinMax(data)
%NORMALIZEMINMAX Scales data columns to [0, 1].
%   [dataNorm, dataMin, dataMax] = normalizeMinMax(data)
%   where
%       dataNorm = (data - dataMin) ./ (dataMax - dataMin).

    dataMin = min(data, [], 1);
    dataMax = max(data, [], 1);
    dataRange = dataMax - dataMin;
    
    % Avoid division by zero
    dataRange(dataRange == 0) = 1e-12;
    
    dataNorm = (data - dataMin) ./ dataRange;
end

