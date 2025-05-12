[XTrain, YTrain] = japaneseVowelsTrainData;
[XTest, YTest] = japaneseVowelsTestData;
numObservations = numel(XTrain);
sequenceLength = cellfun(@(x) size(x,2), XTrain);
classes = categories(YTrain);
numClasses = numel(classes);
inputSize = size(XTrain{1},1);
numHiddenUnits = 100;

layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize', 16, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize', 16, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);
net = trainNetwork(XTrain, YTrain, layers, options);
YPred = classify(net, XTest, 'SequenceLength','longest');
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['Accuracy: ' num2str(accuracy)]);
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix');
[precision, recall, f1] = deal(zeros(numClasses,1));
for i = 1:numClasses
    actual = YTest == classes{i};
    predicted = YPred == classes{i};
    
    tp = sum(actual & predicted);
    fp = sum(~actual & predicted);
    fn = sum(actual & ~predicted);
    
    precision(i) = tp / (tp + fp + eps);
    recall(i) = tp / (tp + fn + eps);
    f1(i) = 2 * (precision(i)*recall(i)) / (precision(i)+recall(i) + eps);
end

table(classes, precision, recall, f1)
Accuracy: 0.95946

