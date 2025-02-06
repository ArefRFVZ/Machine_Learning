% Parameters
M = 10; % Number of membership functions
num_training = 800; % Number of training samples
total_num = 1000;
landa = 0.1; % Stepsize constant

% Preallocation
x_bar = zeros(num_training, M);
g_bar = zeros(num_training, M);
sigma = zeros(num_training, M);
y = zeros(total_num, 1);
u = zeros(total_num, 1);
x = zeros(total_num, 1);
y_hat = zeros(total_num, 1);
f_hat = zeros(total_num, 1);
z = zeros(total_num, 1);
g_u = zeros(total_num, 1);


y(1) = 0;
u(1) = rand;
g_u(1) = 0.6 * sin(pi * u(1)) + 0.3 * sin(3 * pi * u(1)) + 0.1 * sin(5 * pi * u(1));
f_hat(1) = g_u(1);

u_min = -1;
u_max = 1;
h = (u_max - u_min) / (M - 1);

for k = 1:M
    x_bar(1, k) = -1 + h * (k - 1);
    u(1, k) = x_bar(1, k);
    g_bar(1, k) = 0.6 * sin(pi * u(1, k)) + 0.3 * sin(3 * pi * u(1, k)) + 0.1 * sin(5 * pi * u(1, k));
end

sigma(1, 1:M) = (max(u(1, :)) - min(u(1, :))) / M;
x_bar(2, :) = x_bar(1, :); 
g_bar(2, :) = g_bar(1, :); 
sigma(2, :) = sigma(1, :);

x_bar_initial = x_bar(1, :);
sigma_initial = sigma(1, :);
y_bar_initial = g_bar(1, :);

% Training phase with error back-propagation
for q = 2:num_training
    b = 0; a = 0;
    x(q) = -1 + 2 * rand; % Random input
    u(q) = x(q);
    g_u(q) = 0.6 * sin(pi * u(q)) + 0.3 * sin(3 * pi * u(q)) + 0.1 * sin(5 * pi * u(q));
    
    Z = zeros(1, M);
    for r = 1:M
        z(r) = exp(-((x(q) - x_bar(q, r)) / sigma(q, r))^2);
        Z(r) = z(r);
        b = b + z(r);
        a = a + g_bar(q, r) * z(r);
    end
    
    f_hat(q) = a / b; % Output approximation
    
    % Update membership function parameters using gradient descent
    error = g_u(q) - f_hat(q);
    for r = 1:M
        g_bar(q + 1, r) = g_bar(q, r) + landa * Z(r) * error;
        sigma(q + 1, r) = sigma(q, r) + landa * Z(r) * error * ((x(q) - x_bar(q, r))^2 / sigma(q, r)^3);
        x_bar(q + 1, r) = x_bar(q, r) + landa * Z(r) * error * ((x(q) - x_bar(q, r)) / sigma(q, r)^2);
    end
    
    x_bar(q + 1, :) = x_bar(q, :); 
    sigma(q + 1, :) = sigma(q, :); 
    
    y(q + 1) = 0.3 * y(q) + 0.6 * y(q - 1) + g_u(q);
    y_hat(q + 1) = 0.3 * y(q) + 0.6 * y(q - 1) + f_hat(q);
end

x_bar_final = x_bar(num_training, :);
sigma_final = sigma(num_training, :);
g_bar_final = g_bar(num_training, :);

% Test phase
for q = num_training:1000
    b = 0; a = 0;
    x(q) = sin(2 * q * pi / 200);
    u(q) = x(q);
    g_u(q) = 0.6 * sin(pi * u(q)) + 0.3 * sin(3 * pi * u(q)) + 0.1 * sin(5 * pi * u(q));
    
    Z = zeros(1, M);
    for r = 1:M
        z(r) = exp(-((x(q) - x_bar(num_training, r)) / sigma(num_training, r))^2);
        Z(r) = z(r);
        b = b + z(r);
        a = a + g_bar(num_training, r) * z(r);
    end
    
    f_hat(q) = a / b; % Output approximation
    y(q + 1) = 0.3 * y(q) + 0.6 * y(q - 1) + g_u(q);
    y_hat(q + 1) = 0.3 * y(q) + 0.6 * y(q - 1) + f_hat(q);
end

% RMSE Calculation
test_data = num_training+1:total_num;
RMSE = sqrt(mean((y(test_data) - y_hat(test_data)).^2));
fprintf('RMSE for test data: %.4f\n', RMSE);

% Plot results
figure1 = figure('Color', [1 1 1]);
plot(1:1001, y, 'b', 1:1001, y_hat, 'r:', 'Linewidth', 2);
legend('Output of the plant', 'Output of the identification model');
axis([0 1001 -5 5]);
grid on;

% Fuzzy Surface Plot
[x_plot, y_plot] = meshgrid(linspace(u_min, u_max, 100), linspace(u_min, u_max, 100));
z_plot = zeros(size(x_plot));

for i = 1:size(x_plot, 1)
    for j = 1:size(x_plot, 2)
        x_input = x_plot(i, j);
        b = 0; a = 0;
        Z = zeros(1, M);
        for r = 1:M
            z(r) = exp(-((x_input - x_bar(num_training, r)) / sigma(num_training, r))^2);
            Z(r) = z(r);
            b = b + z(r);
            a = a + g_bar(num_training, r) * z(r);
        end
        z_plot(i, j) = a / b;
    end
end

figure2 = figure('Color', [1 1 1]);
surf(x_plot, y_plot, z_plot);
xlabel('Input (x)');
ylabel('Input (y)');
zlabel('Output (f_hat)');
title('Fuzzy Surface of the System');
