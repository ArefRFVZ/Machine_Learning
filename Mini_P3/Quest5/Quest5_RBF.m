clc;
clear all;
close all;

data = readtable('AirQualityUCI.xlsx');
outputData = data.NO2_GT_;
inputData = data{:, setdiff(data.Properties.VariableNames, {'NO2_GT_', 'Date', 'Time'})};

inputData(inputData == -200) = NaN;
outputData(outputData == -200) = NaN;

validRows = ~any(isnan(inputData), 2) & ~isnan(outputData);
inputData = inputData(validRows, :);
outputData = outputData(validRows, :);

inputData = normalize(inputData);

rng(73);
n = size(inputData, 1);
idx = randperm(n);
trainIdx = idx(1:round(0.6*n));
valIdx = idx(round(0.6*n)+1:round(0.8*n));
testIdx = idx(round(0.8*n)+1:end);

X_train = inputData(trainIdx, :);
Y_train = outputData(trainIdx);

X_val = inputData(valIdx, :);
Y_val = outputData(valIdx);

X_test = inputData(testIdx, :);
Y_test = outputData(testIdx);

numRBFNeurons = 15;
net = newrb(X_train', Y_train', 0, 5, numRBFNeurons, 1);
Y_val_pred = sim(net, X_val');
mse_val = mean((Y_val' - Y_val_pred).^2);
fprintf('Validation MSE: %f\n', mse_val);

Y_test_pred = sim(net, X_test');
mse_test = mean((Y_test' - Y_test_pred).^2);
fprintf('Test MSE: %f\n', mse_test);

figure;
subplot(2,1,1);
half_testIdx = 1:round(length(Y_test)/2);
plot(Y_test(half_testIdx), 'b');
hold on;
plot(Y_test_pred(half_testIdx), 'r');
legend('Actual', 'Predicted');
xlabel('Sample Index');
ylabel('NO2(GT)');
title('Test Set: Actual vs Predicted');
subplot(2,1,2);
test_error = Y_test' - Y_test_pred;
plot(test_error(half_testIdx), 'k');
xlabel('Sample Index');
ylabel('Test Error');
title('Test Set: Prediction Error');

figure;
subplot(2,1,1);
half_valIdx = 1:round(length(Y_val)/2);
plot(Y_val(half_valIdx), 'b');
hold on;
plot(Y_val_pred(half_valIdx), 'r');
legend('Actual', 'Predicted');
xlabel('Sample Index');
ylabel('NO2(GT)');
title('Validation Set: Actual vs Predicted');
subplot(2,1,2);
val_error = Y_val' - Y_val_pred;
plot(val_error(half_valIdx), 'k');
xlabel('Sample Index');
ylabel('Validation Error');
title('Validation Set: Prediction Error');

figure;
histfit(test_error, 20, 'normal');
xlabel('Error of NO2(GT)');
ylabel('Probability Density');
title('Histogram of Test Error');
