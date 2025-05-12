load building_dataset.mat
% Load the dataset
[x, t] = building_dataset;

% Create and configure the network
net = fitnet(10);                          % 10 neurons in hidden layer
net.divideParam.trainRatio = 0.7;         % 70% training
net.divideParam.valRatio = 0.15;          % 15% validation
net.divideParam.testRatio = 0.15;         % 15% testing

% Train the network
[net, tr] = train(net, x, t);

% Predict using trained model
y = net(x);

% Display performance
fprintf('Best Training MSE: %.4f\n', tr.best_perf);
fprintf('Best Validation MSE: %.4f\n', tr.best_vperf);
fprintf('Best Test MSE: %.4f\n', tr.best_tperf);
fprintf('Best Epoch: %d\n', tr.best_epoch);

% Plots
plotperform(tr);
plotregression(t, y);
ploterrhist(t - y);
Best Training MSE: 0.0022
Best Validation MSE: 0.0024
Best Test MSE: 0.0025
Best Epoch: 122
plotregression(t, y);
ploterrhist(t - y);
plotperform(tr);
plotperform(tr);
ploterrhist(t - y);
plotregression(t, y);