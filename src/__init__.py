import numpy as np
import pandas as pd
import random

pd.set_option('display.expand_frame_repr', False)


def header(msg):
    print(msg)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights_1 = np.random.rand(self.input.shape[1], 6)
        self.weights_2 = np.random.rand(6, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        # layer1 is hidden layer
        self.layer1 = sigmoid(np.dot(self.input, self.weights_1))
        self.output = sigmoid(np.dot(self.layer1, self.weights_2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights_2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights_1 = np.dot(self.input.T,
                             (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                     self.weights_2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the 153553derivative (slope) of the loss function
        self.weights_1 += d_weights_1
        self.weights_2 += d_weights_2

    def SGD(self, training_data, epochs, mini_batch_size, eta):
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid_output):
    return sigmoid_output*(1-sigmoid_output)


def normalize(data_col):
    output = np.zeros(data_col.shape)

    max_of_row = data_col[0][0]
    min_of_row = data_col[0][0]

    for row in data_col:
        if row[0] > max_of_row:
            max_of_row = row[0]
        if row[0] < min_of_row:
            min_of_row = row[0]

    for index, data in enumerate(data_col):
        output[index] = (data[0] - min_of_row)/(max_of_row - min_of_row)

    return output


use_cols = ['stock_symbol', 'Opening_Price', 'Max_Price', 'Min_Price', 'Closing_Price', 'Next_Day_Closing_Price']

df = pd.read_csv("stocks_data.csv", usecols=use_cols)
data_np = df.to_numpy()

input_x = data_np[:, [1, 2, 3, 4]]
input_x_op = data_np[:, [1]]
input_x_max_p = data_np[:, [2]]
input_x_min_p = data_np[:, [3]]
input_x_cp = data_np[:, [4]]
y = data_np[:, [5]]

normalized_x_op = normalize(input_x_op)
normalized_x_max_p = normalize(input_x_max_p)
normalized_x_min_p = normalize(input_x_min_p)
normalized_x_cp = normalize(input_x_cp)

normalized_x = np.concatenate((normalized_x_op, normalized_x_max_p, normalized_x_min_p, normalized_x_cp), axis=1)
normalized_y = normalize(y)

print(normalized_x.shape)
print(normalized_y)

count = 0
neural_net = NeuralNetwork(normalized_x, normalized_y)
while count < 153553:
    neural_net.feedforward()
    neural_net.backprop()

    if count % 100 == 0:
        print(count)
    count += 1
