import numpy as np

class ActivationLayer:
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.derivative = activation_derivative
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        # Zincir kuralı: Hata * Aktivasyonun Türevi
        return output_gradient * self.derivative(self.input)

# Kullanacağımız fonksiyonlar
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_prime(x): 
    s = sigmoid(x)
    return s * (1 - s)

def relu(x): return np.maximum(0, x)
def relu_prime(x): return (x > 0).astype(float)