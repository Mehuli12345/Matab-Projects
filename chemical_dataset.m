[x, t] = chemical_dataset;
net = fitnet(10); % You can tune number of neurons
[net, tr] = train(net, x, t);
y = net(x);
figure, plotperform(tr);      % Performance Plot
figure, plotregression(t, y); % Regression Plot
figure, ploterrhist(t - y);   % Error Histogram
