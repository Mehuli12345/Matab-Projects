[input, target] = maglev_dataset;
% Define delay parameters and network structure
inputDelays = 1:2;
feedbackDelays = 1:2;
hiddenLayerSize = 10;

% Create NARX network
net = narxnet(inputDelays, feedbackDelays, hiddenLayerSize);
% Prepare data using network structure
[Xs, Xi, Ai, Ts] = preparets(net, input, {}, target);
% Choose training algorithm and set data division ratios
net.trainFcn = 'trainlm'; % Levenberg-Marquardt algorithm

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% Train the NARX neural network
[net, tr] = train(net, Xs, Ts, Xi, Ai);
% Predict output using trained network
Y = net(Xs, Xi, Ai);

% Convert cell arrays to numeric arrays
T = cell2mat(Ts);
Yhat = cell2mat(Y);
% Calculate evaluation metrics
mseVal = mean((T - Yhat).^2);
rmseVal = sqrt(mseVal);
maeVal = mean(abs(T - Yhat));
R2 = 1 - sum((T - Yhat).^2) / sum((T - mean(T)).^2);

% Save results to a text file
fileID = fopen('performance_metrics.txt','w');
fprintf(fileID, 'Mean Squared Error (MSE): %.6f\n', mseVal);
fprintf(fileID, 'Root Mean Squared Error (RMSE): %.6f\n', rmseVal);
fprintf(fileID, 'Mean Absolute Error (MAE): %.6f\n', maeVal);
fprintf(fileID, 'R-squared (R²): %.6f\n', R2);
fclose(fileID);
figure;
plotperform(tr);
figure;
plot(T, 'b'); hold on;
plot(Yhat, 'r--');
legend('Target', 'Predicted');
xlabel('Time Step');
ylabel('Output');
title('Target vs. Predicted Output');
figure;
ploterrhist(T - Yhat);
title('Error Histogram');
figure;
plotregression(T, Yhat, 'Regression');
fprintf('Mean Squared Error (MSE): %.6f\n', mseVal);
fprintf('Root Mean Squared Error (RMSE): %.6f\n', rmseVal);
fprintf('Mean Absolute Error (MAE): %.6f\n', maeVal);
fprintf('R-squared (R²): %.6f\n', R2);
Mean Squared Error (MSE): 0.000001
Root Mean Squared Error (RMSE): 0.000857
Mean Absolute Error (MAE): 0.000288
R-squared (R²): 1.000000