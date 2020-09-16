%  This classifies images of handwritten digits (0-6) into class (0-6).
%  Full explanation available at: https://boxofcubes.wordpress.com/2020/09/16/multiclass-logistic-regression/

clear; close all; clc;

%  Upload the data, split into training and testing sets
load number_images.txt
x_all = number_images;
load number_labels.txt
y_all = number_labels;

full = [y_all, x_all];
[m, n] = size(full);
p = .7;
i = randperm(m);

train_full = full(i(1:round(p*m)), :);
y_train = train_full(:, 1);
x_train = train_full(:, 2:n);

test_full = full(i(round(p*m)+1:end), :);
y_test = test_full(:, 1);
x_test = test_full(:, 2:n);

fprintf('\nData uploaded and split into training and testing sets.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Make design matrix
X_train = [ones(length(y_train), 1), x_train];

%  Declare relevant functions
function g = sigmoid(z)
g = 1 ./ (1 + e .^ -z);
end;

function [J, grad] = regularizedCost(theta, X, y, lambda)
m = length(y);
grad = zeros(size(theta));
predictions = sigmoid(X * theta); 
t = theta(2:length(theta));
J = (-y' * log(predictions) - (1-y)' * log(1 - predictions))/m + (lambda/(2*m))*sum(t.^2);
grad = ((predictions - y)' * X / m)' + lambda .* theta .* [0; ones(length(theta)-1, 1)] ./ m;
end;

function [all_theta] = findAllTheta(X, y, classes,lambda)
[m, n] = size(X);
all_theta = zeros(classes, n);
for class=1:classes  
  first_theta = zeros(n, 1); 
  options = optimset('GradObj', 'on', 'Maxiter', 50);
  [theta] = fminunc (@(t)(regularizedCost(t, X, (y == class), lambda)), first_theta, options); 
  all_theta(class, :) = theta;
end;
end;

%  Run fminunc
all_theta = findAllTheta(X_train, y_train, n, .1);

fprintf('\nFound optimal values for theta.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Declare predict() function
function p = predict(all_theta, X)
m = size(X, 1);
classes = size(all_theta, 1);
p = zeros(size(X,1)-1, 1);
all_predictions = zeros(m, size(all_theta, 1));
all_predictions = sigmoid(X * all_theta');
[highest_probability, class] = max(all_predictions, [], 2);
p = class;
end;

%  Predict numbers for training images and get accuracy
p = predict(all_theta, X_train)
accuracy = sum(y_train==p) / length(y_train)

fprintf('\nThose are the predictions of which number the training images depict, and the accuracy rate of the model.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Make predictions on testing images and get accuracy
X_test = [ones(length(y_test), 1), x_test];
p_test = predict(all_theta, X_test)
ac_test = sum(y_test==p_test) / length(y_test)

fprintf('\nThose are the predictions of which number the testing images depict, and the accuracy rate of the model.\n');
