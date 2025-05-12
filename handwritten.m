% Load digit dataset
[XTrain, YTrain] = digitTrain4DArrayData;
[XTest, YTest] = digitTest4DArrayData;

% Define CNN architecture
layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(64)
    reluLayer

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% Set training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'ValidationData', {XTest, YTest}, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(XTrain, YTrain, layers, options);

% Predict and evaluate
YPred = classify(net, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['Test Accuracy: ', num2str(accuracy)])

% Confusion matrix
figure
confusionchart(YTest, YPred);
title('Confusion Matrix')

% ROC curve (one-vs-all for multi-class)
scores = predict(net, XTest); % Get predicted probabilities

% Convert true labels to numeric values
trueLabels = double(YTest);

% Plot ROC curve for each class
figure
hold on
for i = 1:10  % Since we have 10 classes (digits 0-9)
    [X, Y, ~, AUC] = perfcurve(trueLabels, scores(:, i), i, 'XCrit', 'fpr', 'YCrit', 'tpr');
    plot(X, Y, 'DisplayName', ['Class ' num2str(i-1) ' (AUC = ' num2str(AUC) ')']);
end
hold off

xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Curve for Multi-Class Classification')
legend show

% Compute confusion matrix for precision, recall, F1-score
cm = confusionmat(trueLabels, YPred);

% Initialize variables for metrics
precision = zeros(1, 10);
recall = zeros(1, 10);
f1Score = zeros(1, 10);

% Calculate precision, recall, and F1-score for each class
for i = 1:10
    TP = cm(i, i);  % True positives for class i
    FP = sum(cm(:, i)) - TP;  % False positives for class i
    FN = sum(cm(i, :)) - TP;  % False negatives for class i
    
    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Display precision, recall, and F1-score for each class
disp('Precision for each class:');
disp(precision);
disp('Recall for each class:');
disp(recall);
disp('F1-score for each class:');
disp(f1Score);

% Average precision, recall, and F1-score (macro-average)
avgPrecision = mean(precision);
avgRecall = mean(recall);
avgF1Score = mean(f1Score);

disp(['Average Precision: ', num2str(avgPrecision)]);
disp(['Average Recall: ', num2str(avgRecall)]);
disp(['Average F1-score: ', num2str(avgF1Score)]);

Test Accuracy: 0.99