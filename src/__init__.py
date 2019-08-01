import numpy as np
import pandas as pd
import random

pd.set_option('display.expand_frame_repr', False)


def header(msg):
    print(msg)


class NeuralNetwork:
    accuracy = 0

    def __init__(self, x, y):
        self.input = x
        self.weights_1 = np.random.randn(self.input.shape[1], 5)
        self.weights_2 = np.random.randn(5, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        # layer1 is hidden layer
        self.layer1 = sigmoid(np.dot(self.input, self.weights_1))
        self.output = sigmoid(np.dot(self.layer1, self.weights_2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights_2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights_1 = np.dot(self.input.T,
                             (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                     self.weights_2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights_1 += d_weights_1
        self.weights_2 += d_weights_2

    def evaluate(self, testing_x, testing_y):
        self.input = testing_x
        self.y = testing_y
        self.feedforward()
        for index in range(self.y.size):
            if int(self.output[index]) == self.y[index]:
                NeuralNetwork.accuracy += 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)


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
        output[index] = (data[0] - min_of_row) / (max_of_row - min_of_row)

    return output


use_cols = ['stock_symbol', 'Total_Transactions', 'Total_Traded_Shares', 'Total_Traded_Amount',
            'Opening_Price', 'Max_Price', 'Min_Price', 'Closing_Price', 'Next_Day_Closing_Price']

df = pd.read_csv("stocks_data.csv", usecols=use_cols)
data_np = df.to_numpy()

input_x = data_np[:, [1, 2, 3, 4, 5, 6, 7]]
input_x_tt = data_np[:, [1]]
input_x_tts = data_np[:, [2]]
input_x_tta = data_np[:, [3]]
input_x_op = data_np[:, [4]]
input_x_max_p = data_np[:, [5]]
input_x_min_p = data_np[:, [6]]
input_x_cp = data_np[:, [7]]

closing_price = data_np[:, [7]]
next_day_closing_price = data_np[:, [8]]
diff = closing_price - next_day_closing_price
y = (diff < 0).astype(int)

normalized_x_tt = normalize(input_x_tt)
normalized_x_tts = normalize(input_x_tts)
normalized_x_tta = normalize(input_x_tta)
normalized_x_op = normalize(input_x_op)
normalized_x_max_p = normalize(input_x_max_p)
normalized_x_min_p = normalize(input_x_min_p)
normalized_x_cp = normalize(input_x_cp)

normalized_x = np.concatenate((normalized_x_tt, normalized_x_tts, normalized_x_tta,
                               normalized_x_op, normalized_x_max_p, normalized_x_min_p, normalized_x_cp), axis=1)

split = 115164
train_x, train_y = normalized_x[:split, :], y[:split, :]
test_x, test_y = normalized_x[split:, :], y[split:, :]

neural_net = NeuralNetwork(train_x, train_y)
for _ in range(1500):
    neural_net.feedforward()
    neural_net.backprop()

neural_net.evaluate(test_x, test_y)
print(NeuralNetwork.accuracy/test_y.size)
