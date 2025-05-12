load bodyfat_dataset.mat
[x, t] = bodyfat_dataset;
net = fitnet(10);                     % 10 neurons in hidden layer
net.divideParam.trainRatio = 0.7;     % 70% training
net.divideParam.valRatio = 0.15;      % 15% validation
net.divideParam.testRatio = 0.15;     % 15% testing

[net, tr] = train(net, x, t);         % Train the network
y = net(x);                           % Predict using trained model
fprintf('Best Training MSE: %.4f\n', tr.best_perf);
fprintf('Best Validation MSE: %.4f\n', tr.best_vperf);
fprintf('Best Test MSE: %.4f\n', tr.best_tperf);
fprintf('Best Epoch: %d\n', tr.best_epoch);
Best Training MSE: 10.3011
Best Validation MSE: 26.6393
Best Test MSE: 53.7680
Best Epoch: 9
plotperform(tr);
plotregression(t, y);
ploterrhist(t - y)