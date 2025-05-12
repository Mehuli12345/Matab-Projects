% Load pollution data (example: use your own data loading method)
[X, T] = pollution_dataset;  % Replace with your actual pollution dataset

% Prepare the TDNN model
inputDelays = 1:2;  % Adjust based on system dynamics
hiddenLayerSize = 10;
net = timedelaynet(inputDelays, hiddenLayerSize);

% Prepare data for training
[Xs, Xi, Ai, Ts] = preparets(net, X, T);

% Divide data for training, validation, testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the network
[net, tr] = train(net, Xs, Ts, Xi, Ai);

% Test the network
Y = net(Xs, Xi, Ai);
perf = perform(net, Ts, Y);

% Calculate metrics
targets = cell2mat(Ts);
outputs = cell2mat(Y);
mseVal = mean((targets - outputs).^2);
rmseVal = sqrt(mseVal);
maeVal = mean(abs(targets - outputs));
R2 = 1 - sum((targets - outputs).^2) / sum((targets - mean(targets)).^2);

% Display results
fprintf('MSE: %.6f\nRMSE: %.6f\nMAE: %.6f\nR²: %.6f\n', mseVal, rmseVal, maeVal, R2);

% Plot results
figure, plotperform(tr);
figure, plotregression(Ts, Y);
figure, ploterrhist(cell2mat(Ts) - cell2mat(Y));
figure, plot(cell2mat(Ts), 'b'); hold on;
plot(cell2mat(Y), 'r'); legend('Target', 'Output');
title('Pollution Prediction: Target vs Output');
MSE: 22.822705