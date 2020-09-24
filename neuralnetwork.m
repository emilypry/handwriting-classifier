% This is a neural network that classifies images of handwritten digits, 0-5.
%  Full explanation available at: https://boxofcubes.wordpress.com/2020/09/16/a-neural-network/

clear; close all; clc;

%  Upload the data, split into training and testing sets
load number_images.txt
x_all = number_images;
load number_labels.txt
y_all = number_labels;
%  Get rid of class 0 (for indexing reasons); y=1 now means image is of a 0, 
%  y=2 means image is of a 1, etc.
y_all += 1;

%  Split data into training and testing sets
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

%  Make design matrix
X_train = [ones(length(y_train),1), x_train];

fprintf('\nData uploaded and split into training and testing sets.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Declare relevant functions
function g = sigmoid(z)
g = 1 ./ (1 + e .^ -z);
end;

function g = sigmoidGradient(z)
g = sigmoid(z) .* (1-sigmoid(z));
end;

function unrolled = randomlyInitialize(X, hidden_nodes, classes)
% Works for a 3-layer network (only 1 hidden layer)
% Returns an unrolled vector of random Theta1 and Theta2 matrices
% X = design matrix
% hidden_nodes = number of hidden nodes, excluding bias unit
% classes = number of classes/output nodes
E = .12;
Theta1 = rand(hidden_nodes, size(X,2)) * (2*E) - E;
Theta2 = rand(classes, hidden_nodes+1) * (2*E) - E;
unrolled = [Theta1(:); Theta2(:)];
end;

function [J, grad] = cost(unrolled_theta, hidden_nodes, classes, X, y, lambda)
% Works for a 3-layer network (only 1 hidden layer)
% Theta1, Theta2 = matrices of theta for hidden and output layers
% input_nodes = number of input nodes, excluding bias unit
% hidden_nodes = number of hidden nodes, excluding bias unit
% classes = number of classes/output nodes
% X = design matrix

% FEEDFORWARD PROPAGATION
[m, all_features] = size(X);

% Roll unrolled theta vector back into matrices
Theta1 = reshape(unrolled_theta(1:hidden_nodes * all_features), hidden_nodes, all_features);
Theta2 = reshape(unrolled_theta(hidden_nodes * all_features + 1:end), classes, hidden_nodes+1);

% Find hidden layer values, add bias unit
hidden_layer_values = sigmoid(X * Theta1');
hidden_layer_values = [ones(m, 1), hidden_layer_values]; 

% Find output layer values
output_layer_values = sigmoid(hidden_layer_values * Theta2');

% Make a matrix to convert all y values to vectors
Y = zeros(m, classes);

% Make a vector to store the cost of the network for each example
costs = zeros(m,1);

% COST
for i=1:m
  % Mark the column/class for each y with a 1
  Y(i,y(i)) = 1;

  % Get the cost for that example
  costs(i) = log(output_layer_values(i,:))*(-Y(i,:))' - log(1-output_layer_values(i,:))*(1-Y(i,:))';
end;

% Get the cost of the network for whole set of examples
J = 1/m * sum(costs);
% Add the regularization term to the cost to prevent overfitting
J += lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% BACK PROPAGATION 
% Make matrices to store gradients for Thetas
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
 
for i = 1:m
  input = X(i,:)';

  z2 = Theta1 * input;
  hidden_layer_values = [1; sigmoid(z2)];

  z3 = Theta2 * hidden_layer_values;
  output_layer_values = sigmoid(z3);

  output_error = output_layer_values - Y(i,:)';

  hidden_error = (Theta2' * output_error).*[1; sigmoidGradient(z2)];
  hidden_error = hidden_error(2:end);

  Theta1_grad = Theta1_grad + hidden_error * input';
  Theta2_grad = Theta2_grad + output_error * hidden_layer_values';
end;

Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;

% Regularize the gradients
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);

% Unroll the gradient matrices
grad = [Theta1_grad(:); Theta2_grad(:)];
end;

fprintf('\nCost function has been declared.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Get an initial set of theta values and check cost
initial_theta = randomlyInitialize(X_train, 750, 6);
[J, grad] = cost(initial_theta, 750, 6, X_train, y_train, .1, 1);
J

fprintf('\nThat is the cost of your network for randomly initialized values of theta.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Double check your gradients to make sure they look right
%  The following code is primarily due to Andrew Ng ---->
function numgrad = computeNumericalGradient(J, theta)
numgrad = zeros(size(theta));
perturb = zeros(size(theta));
e = 1e-4;
for p = 1:numel(theta)
    % Set perturbation vector
    perturb(p) = e;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
end
end

function W = debugInitializeWeights(fan_out, fan_in)
% Set W to zeros
W = zeros(fan_out, 1 + fan_in);
% Initialize W using "sin", this ensures that W is always of the same
% values and will be useful for debugging
W = reshape(sin(1:numel(W)), size(W)) / 10;
end

function checkNNGradients(lambda)
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;

% We generate some 'random' test data
Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
% Reusing debugInitializeWeights to generate X
X  = debugInitializeWeights(m, input_layer_size - 1);
X = [ones(size(X,1),1), X];
y  = 1 + mod(1:m, num_labels)';

% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];

% Short hand for cost function
costFunc = @(p) cost(p, hidden_layer_size, num_labels, X, y, lambda);

[cost, grad] = costFunc(nn_params);
numgrad = computeNumericalGradient(costFunc, nn_params);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad, grad])
fprintf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);
fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);
end
%  <--------

%  Check your gradients
checkNNGradients(.1)

fprintf(['\nAs long as the gradients through cost() (left) look similar to \n' ...
'the numerically derived gradients (right), you should be good. \n' ...
'Ready to train your network?']);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Declare the trainNetwork() function
function [all_theta, cost] = trainNetwork(initial_theta, hidden_nodes, classes, X, y, lambda)
options = optimset('MaxIter', 50);
costFunction = @(t) cost(t, hidden_nodes, classes, X, y, lambda)
[all_theta, cost] = fmincg(costFunction, initial_theta, options);
end

%  Train the network
[all_theta, cost] = trainNetwork(initial_theta, 750, 6, X_train, y_train, .1);

Theta1 = reshape(all_theta(1: 750 * size(X_train,2)), 750, size(X_train,2));
Theta2 = reshape(all_theta(1 + 750 * size(X_train,2):end), 6, 750+1);

fprintf('\nNetwork has been trained. Let us see its predictions.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Declare predict() function
function p = predict(Theta1, Theta2, X)
% X = design matrix
m = size(X, 1);
classes = size(Theta2, 1);
p = zeros(m, 1);
hidden_layer_values = sigmoid(X * Theta1');
hidden_layer_values = [ones(m,1), hidden_layer_values];
output_layer_values = sigmoid(hidden_layer_values * Theta2');
[value, p] = max(output_layer_values, [], 2);
end

function [J, grad] = cost(unrolled_theta, hidden_nodes, classes, X, y, lambda)
% Works for a 3-layer network (only 1 hidden layer)
% Theta1, Theta2 = matrices of theta for hidden and output layers
% input_nodes = number of input nodes, excluding bias unit
% hidden_nodes = number of hidden nodes, excluding bias unit
% classes = number of classes/output nodes
% X = design matrix

% FEEDFORWARD PROPAGATION
[m, all_features] = size(X);

% Roll unrolled theta vector back into matrices
Theta1 = reshape(unrolled_theta(1:hidden_nodes * all_features), hidden_nodes, all_features);
Theta2 = reshape(unrolled_theta(hidden_nodes * all_features + 1:end), classes, hidden_nodes+1);

% Find hidden layer values, add bias unit
hidden_layer_values = sigmoid(X * Theta1');
hidden_layer_values = [ones(m, 1), hidden_layer_values]; 

% Find output layer values
output_layer_values = sigmoid(hidden_layer_values * Theta2');

% Make a matrix to convert all y values to vectors
Y = zeros(m, classes);

% Make a vector to store the cost of the network for each example
costs = zeros(m,1);

% COST
for i=1:m
  % Mark the column/class for each y with a 1
  Y(i,y(i)) = 1;

  % Get the cost for that example
  costs(i) = log(output_layer_values(i,:))*(-Y(i,:))' - log(1-output_layer_values(i,:))*(1-Y(i,:))';
end;

% Get the cost of the network for whole set of examples
J = 1/m * sum(costs);
% Add the regularization term to the cost to prevent overfitting
J += lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% BACK PROPAGATION 
% Make matrices to store gradients for Thetas
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
 
for i = 1:m
  input = X(i,:)';

  z2 = Theta1 * input;
  hidden_layer_values = [1; sigmoid(z2)];

  z3 = Theta2 * hidden_layer_values;
  output_layer_values = sigmoid(z3);

  output_error = output_layer_values - Y(i,:)';

  hidden_error = (Theta2' * output_error).*[1; sigmoidGradient(z2)];
  hidden_error = hidden_error(2:end);

  Theta1_grad = Theta1_grad + hidden_error * input';
  Theta2_grad = Theta2_grad + output_error * hidden_layer_values';
end;

Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;

% Regularize the gradients
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);

% Unroll the gradient matrices
grad = [Theta1_grad(:); Theta2_grad(:)];
end;

%  Predict and get costs for training and testing sets
p_train = predict(Theta1, Theta2, X_train);
accuracy_train = sum(y_train==p_train) / length(y_train)
cost_train = cost(all_theta, 750, 6, X_train, y_train, 0)

X_test = [ones(length(y_test),1), x_test];
p_test = predict(Theta1, Theta2, X_test);
accuracy_test = sum(y_test==p_test) / length(y_test)
cost_test = cost(all_theta, 750, 6, X_test, y_test, 0)


fprintf('\nThere are the predictions and cost for the training and testing sets.\n');
