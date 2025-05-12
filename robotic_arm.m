% Step 1: Load dataset
[X, T] = robotarm_dataset;

% Step 2: Convert to time series format
inputSeries = X;      % Torque inputs
targetSeries = T;     % Joint angles

% Step 3: Create NARX network
inputDelays = 1:2;     % Input delays
feedbackDelays = 1:2;  % Feedback delays
hiddenLayerSize = 10;
net = narxnet(inputDelays, feedbackDelays, hiddenLayerSize);

% Step 4: Prepare data for training
[inputs, inputStates, layerStates, targets] = preparets(net, inputSeries, {}, targetSeries);

% Step 5: Data division for training, validation, testing
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

% Step 6: Train the network
[net, tr] = train(net, inputs, targets, inputStates, layerStates);

% Step 7: Test network
outputs = net(inputs, inputStates, layerStates);
performance = perform(net, targets, outputs);

% Step 8: Error metrics
targetVec = cell2mat(targets);
outputVec = cell2mat(outputs);

mseVal  = mean((targetVec - outputVec).^2);
rmseVal = sqrt(mseVal);
maeVal  = mean(abs(targetVec - outputVec));
R2 = 1 - sum((targetVec - outputVec).^2) / sum((targetVec - mean(targetVec)).^2);

% Step 9: Plot results
figure; plotperform(tr);              title('Training Performance');
figure; plotresponse(targets, outputs); title('Regression Plot');
figure; ploterrhist(targetVec - outputVec); title('Error Histogram');
figure; plot(cell2mat(targets)', 'b'); hold on; 
plot(cell2mat(outputs)', 'r--'); legend('Target', 'Predicted');
title('Time Series Prediction');