import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        # Xavier Initialization
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate):
        # 1. Ağırlıklar için hata (dW)
        weights_gradient = np.dot(self.input.T, output_gradient)
        # 2. Girdi için hata (Bir önceki katmana aktarılacak olan)
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # Ağırlıkları güncelle (Gradient Descent)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        
        return input_gradient