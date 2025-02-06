clc;
clear all;
close all;

% Load the data from the Excel file
data = readtable('AirQualityUCI.xlsx');

% Extract the NO2(GT) column as the output
outputData = data.NO2_GT_;

% Extract the input features (all columns except NO2(GT), Date, and Time)
inputData = data{:, setdiff(data.Properties.VariableNames, {'NO2_GT_', 'Date', 'Time'})};

% Handle missing values (replace -200 with NaN)
inputData(inputData == -200) = NaN;
outputData(outputData == -200) = NaN;

% Remove rows with missing values
validRows = ~any(isnan(inputData), 2) & ~isnan(outputData);
inputData = inputData(validRows, :);
outputData = outputData(validRows, :);

% Manually calculate mean (mu) and standard deviation (sigma) for input and output data
inputMu = mean(inputData, 1);  % Mean of each column (1x12 array)
inputSigma = std(inputData, 0, 1);  % Standard deviation of each column (1x12 array)

outputMu = mean(outputData, 1);  % Mean of output data (scalar)
outputSigma = std(outputData, 0, 1);  % Standard deviation of output data (scalar)

% Normalize the input and output data using MATLAB's normalize function
inputDataNorm = normalize(inputData);  % Normalize input data
outputDataNorm = normalize(outputData);  % Normalize output data

% Split the data into training, testing, and validation sets
rng(73);  % For reproducibility
n = size(inputDataNorm, 1);
trainIndices = 1:round(0.6*n);
testIndices = round(0.6*n)+1:round(0.8*n);
valIndices = round(0.8*n)+1:n;

trainInput = inputDataNorm(trainIndices, :);
trainOutput = outputDataNorm(trainIndices, :);

testInput = inputDataNorm(testIndices, :);
testOutput = outputDataNorm(testIndices, :);

valInput = inputDataNorm(valIndices, :);
valOutput = outputDataNorm(valIndices, :);

% Generate the initial FIS structure using genfis2
radius = 0.5;  % Adjust this parameter as needed
in_fis = genfis2(trainInput, trainOutput, radius);

% Plot membership functions of input 1 and 8 BEFORE training
figure;
subplot(2, 1, 1);
plotmf(in_fis, 'input', 1);  % Membership functions for input 1
title('Membership Functions for Input 1 (Before Training)');
xlabel('Input 1');
ylabel('Degree of Membership');

subplot(2, 1, 2);
plotmf(in_fis, 'input', 8);  % Membership functions for input 8
title('Membership Functions for Input 8 (Before Training)');
xlabel('Input 8');
ylabel('Degree of Membership');

% Train the ANFIS model and capture training/checking error
epochs = 100;  % Number of epochs
[out_fis, trainError, stepSize,~, valError] = anfis([trainInput, trainOutput], in_fis, epochs, [1, 1, 1, 1], [valInput, valOutput]);

% Plot membership functions of input 1 and 8 AFTER training
figure;
subplot(2, 1, 1);
plotmf(out_fis, 'input', 1);  % Membership functions for input 1
title('Membership Functions for Input 1 (After Training)');
xlabel('Input 1');
ylabel('Degree of Membership');

subplot(2, 1, 2);
plotmf(out_fis, 'input', 8);  % Membership functions for input 8
title('Membership Functions for Input 8 (After Training)');
xlabel('Input 8');
ylabel('Degree of Membership');

% Evaluate the model on the training set
predictedTrainOutputNorm = evalfis(trainInput, out_fis);

% Denormalize the predicted and actual outputs for the training set
predictedTrainOutputDenorm = (predictedTrainOutputNorm * outputSigma) + outputMu;
trainOutputDenorm = (trainOutput * outputSigma) + outputMu;

% Calculate RMSE and MSE for the training set
trainErrorDenorm = predictedTrainOutputDenorm - trainOutputDenorm;
trainRMSE = sqrt(mean(trainErrorDenorm.^2));
trainMSE = mean(trainErrorDenorm.^2);

% Evaluate the model on the validation set
predictedValOutputNorm = evalfis(valInput, out_fis);

% Denormalize the predicted and actual outputs for the validation set
predictedValOutputDenorm = (predictedValOutputNorm * outputSigma) + outputMu;
valOutputDenorm = (valOutput * outputSigma) + outputMu;

% Calculate RMSE and MSE for the validation set
valErrorDenorm = predictedValOutputDenorm - valOutputDenorm;
valRMSE = sqrt(mean(valErrorDenorm.^2));
valMSE = mean(valErrorDenorm.^2);

% Evaluate the model on the test set
predictedOutputNorm = evalfis(testInput, out_fis);

% Denormalize the predicted and actual outputs for the test set
predictedOutputDenorm = (predictedOutputNorm * outputSigma) + outputMu;
testOutputDenorm = (testOutput * outputSigma) + outputMu;

% Calculate RMSE and MSE for the test set
testErrorDenorm = predictedOutputDenorm - testOutputDenorm;
testRMSE = sqrt(mean(testErrorDenorm.^2));
testMSE = mean(testErrorDenorm.^2);

% Report denormalized RMSE and MSE for training, validation, and test sets
fprintf('Training Set:\n');
fprintf('  Denormalized RMSE: %.4f\n', trainRMSE);
fprintf('  Denormalized MSE: %.4f\n', trainMSE);

fprintf('Validation Set:\n');
fprintf('  Denormalized RMSE: %.4f\n', valRMSE);
fprintf('  Denormalized MSE: %.4f\n', valMSE);

fprintf('Test Set:\n');
fprintf('  Denormalized RMSE: %.4f\n', testRMSE);
fprintf('  Denormalized MSE: %.4f\n', testMSE);

% Plot the training and checking error over epochs
figure;
plot(1:epochs, trainError, 'b', 'LineWidth', 1.5);
hold on;
plot(1:epochs, valError, 'r', 'LineWidth', 1.5);
legend('Training Error', 'Validation Error');
xlabel('Epochs');
ylabel('Error');
title('Training and Validation Error over Epochs');
grid on;
hold off;

% Plot the actual vs predicted output (denormalized) for the test set
figure;
plot(testOutputDenorm, 'b');
hold on;
plot(predictedOutputDenorm, 'r');
legend('Actual Output', 'Predicted Output');
xlabel('Sample Index');
ylabel('NO2(GT)');
title('ANFIS Regression Results (genfis2) - Denormalized');
hold off;

% Plot the histogram of the test error (denormalized)
figure;
histfit(testErrorDenorm, 20, 'normal');
xlabel('Test Error (Denormalized)');
ylabel('Frequency');
title('Histogram of Test Error (Denormalized)');
grid on;