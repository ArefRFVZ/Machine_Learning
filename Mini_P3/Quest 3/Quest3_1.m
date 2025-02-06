
!gunzip ballbeam.dat.Z
	load ballbeam.dat
input_data=ballbeam(:,1);
output_data=ballbeam(:,2);

num_samples = length(input_data);

random_indices = randperm(num_samples);

train_ratio = 0.85;
num_train = round(train_ratio * num_samples);

train_indices = random_indices(1:num_train);
test_indices = random_indices(num_train+1:end);

input_train = input_data(train_indices);
output_train = output_data(train_indices);

input_test = input_data(test_indices);
output_test = output_data(test_indices);

training_data = [input_train, output_train];

fis = genfis2(input_train, output_train, 0.15);

MaxEpoch = 250;
ErrorGoal = 0;
InitialStepSize = 0.01;
StepSizeDecreaseRate = 0.9;
StepSizeIncreaseRate = 1.1;
TrainOptions = [MaxEpoch, ErrorGoal, InitialStepSize, StepSizeDecreaseRate, StepSizeIncreaseRate];

DisplayInfo = true;
DisplayError = true;
DisplayStepSize = true;
DisplayFinalResult = true;
DisplayOptions = [DisplayInfo, DisplayError, DisplayStepSize, DisplayFinalResult];

OptimizationMethod = 1;

[fis, trainError] = anfis(training_data, fis, TrainOptions, DisplayOptions, [], OptimizationMethod);

TrainOutputs = evalfis(input_train, fis);
TrainErrors = output_train - TrainOutputs;
TrainMSE = mean(TrainErrors(:).^2);
TrainRMSE = sqrt(TrainMSE);

PlotResults(output_train, TrainOutputs, 'Train Data');

TestOutputs = evalfis(input_test, fis);
TestErrors = output_test - TestOutputs;
TestMSE = mean(TestErrors(:).^2);
TestRMSE = sqrt(TestMSE);

PlotResults(output_test, TestOutputs, 'Test Data');

disp(['Train RMSE: ', num2str(TrainRMSE)]);
disp(['Test RMSE: ', num2str(TestRMSE)]);

figure;
plotmf(fis, 'input', 1); 
title('Input Membership Functions for Input 1');

figure;
[X, Y] = meshgrid(linspace(min(input_train), max(input_train), 100));
Z = evalfis([X(:)], fis); 
Z = reshape(Z, size(X));
surf(X, Z);
xlabel('Input');
ylabel('Fuzzy Output');
zlabel('Output');
title('Surface Plot of Fuzzy Inference System');

ruleview(fis); 

figure;
plot(1:MaxEpoch, trainError);
title('Convergence Plot (Training Error over Epochs)');
xlabel('Epoch');
ylabel('Training Error');

figure;
bar([TrainRMSE, TestRMSE]);
set(gca, 'xticklabel', {'Train RMSE', 'Test RMSE'});
title('RMSE Comparison (Train vs Test)');
ylabel('RMSE');

function PlotResults(targets, outputs, Name)
    errors = targets - outputs;
    RMSE = sqrt(mean(errors(:).^2));
    error_mean = mean(errors(:));
    error_std = std(errors(:));
    
    figure;
    plot(targets, 'k');
    hold on;
    plot(outputs, 'r--');
    legend('Target', 'Output');
    title([Name ': Target vs Output']);
    
    figure;
    plot(errors);
    legend('Error');
    title([Name ': Errors']);
    xlabel('Sample Index');
    ylabel('Error');
    
    figure;
    histfit(errors);
    title([Name ': Error Distribution']);
    xlabel('Error');
    ylabel('Frequency');
    legend('Error Histogram', 'Fitted Curve');
end
