class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, learning_rate):
        for i in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # 1. Forward Pass (İleri besleme)
                output = self.predict(x)
                
                # 2. Error Calculation (Hata ölçümü)
                error += mse(y, output)
                
                # 3. Backward Pass (Geriye yayılım)
                gradient = mse_prime(y, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)
            
            if (i+1) % 100 == 0:
                print(f"Epoch {i+1}/{epochs}, Error: {error/len(x_train)}")