from random import shuffle
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image


class CustomMLP:
    def __init__(self, hidden_layers, neurons_per_layer, activation_function, learning_rate, epochs):
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_size = None
        self.output_size = None
        # Initialize weights and biases
        self.weights, self.biases = self.initialize_weights_and_biases()
        self.data, self.labels = self.load_and_preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=42)

        self.model = self.build_model()

    def load_and_preprocess_data(self):
        data = []
        labels = []
        input_size = None

        # Define a function to extract the expression label from the file name
        def extract_expression_label(file_name):
            return file_name.split('_')[1]

        # Load data and labels
        dataset_dir = 'C:/Users/admin/Desktop/Facial-Recognition-System/data/yalefaces_resized'
        for file_name in os.listdir(dataset_dir):
            img_path = os.path.join(dataset_dir, file_name)
            if img_path.lower().endswith('.png'):
                img = Image.open(img_path).convert('L')
                img_array = np.array(img)
                img_array = img_array.astype('float32') / 255.0
                data.append(img_array.flatten())
                labels.append(extract_expression_label(file_name))

        if not data:
            raise ValueError("No valid images found in the dataset directory.")

        print(f"Number of images loaded: {len(data)}")

        if data:
            input_size = data[0].shape[0]
            print(f"Input size: {input_size}")

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        data = np.array(data)
        labels = np.array(labels)

        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        self.input_size = input_size
        self.output_size = len(np.unique(labels))
        print(f"Output size: {self.output_size}")
        return data, labels

    def initialize_weights_and_biases(self):
        weights = []
        biases = []
        input_size = self.input_size
        for _ in range(self.hidden_layers):
            weights_layer = np.random.randn(
                input_size, self.neurons_per_layer) * 0.01
            biases_layer = np.zeros((1, self.neurons_per_layer))
            weights.append(weights_layer)
            biases.append(biases_layer)
            input_size = self.neurons_per_layer
        weights_output_layer = np.random.randn(
            input_size, self.output_size) * 0.01
        biases_output_layer = np.zeros((1, self.output_size))
        weights.append(weights_output_layer)
        biases.append(biases_output_layer)
        return weights, biases

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, X):
        activations = []
        inputs = X
        for i in range(self.hidden_layers):
            # Calculate layer's linear output
            z = np.dot(inputs, self.weights[i]) + self.biases[i]
            # Apply activation function (ReLU)
            activations.append(self.relu(z))
            inputs = activations[-1]
        # Output layer
        logits = np.dot(inputs, self.weights[-1]) + self.biases[-1]
        # Apply softmax for multiclass classification
        output = self.softmax(logits)
        activations.append(output)
        return activations

    def compute_loss(self, y_pred, y_true):
        # Compute categorical cross-entropy loss
        m = y_pred.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward_propagation(self, X, y, activations):
        m = X.shape[0]
        gradients = []
        # Compute gradients for output layer
        output_error = activations[-1]
        output_error[range(m), y] -= 1
        output_error /= m
        gradients.append(output_error)
        # Backpropagate error to hidden layers
        for i in range(self.hidden_layers, 0, -1):
            print("Layer {}: Weights Shape: {}, Gradients Shape: {}".format(
                i, self.weights[i].shape, gradients[-1].shape))
            error = np.dot(gradients[-1], self.weights[i].T)

            # Apply derivative of ReLU activation function
            error[activations[i - 1] <= 0] = 0
            gradients.append(error)
        # Reverse gradients for consistency
        gradients.reverse()
        return gradients

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(self.input_size,)))
        for _ in range(self.hidden_layers):
            model.add(tf.keras.layers.Dense(self.neurons_per_layer,
                      activation=self.activation_function))
        model.add(tf.keras.layers.Dense(
            self.output_size, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def update_weights_and_biases(self, gradients, activations):
        m = activations[0].shape[0]
        for i in range(self.hidden_layers + 1):
            activations_transpose = activations[i].T
            gradients_transpose = gradients[i].T
            # Update weights and biases using gradient descent
            self.weights[i] -= self.learning_rate * \
                np.dot(activations_transpose, gradients_transpose)
            self.biases[i] -= self.learning_rate * \
                np.sum(gradients[i], axis=0, keepdims=True) / m

    def fit(self):
        for epoch in range(self.epochs):
            activations = self.forward_propagation(self.X_train)
            loss = self.compute_loss(activations[-1], self.y_train)
            gradients = self.backward_propagation(
                self.X_train, self.y_train, activations)
            self.update_weights_and_biases(gradients, activations)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def test(self):
        # Forward propagation for test data
        activations = self.forward_propagation(self.X_test)
        # Get predicted labels
        predicted_labels = np.argmax(activations[-1], axis=1)
        # Calculate accuracy
        accuracy = np.mean(predicted_labels == self.y_test)
        print(f'Test Accuracy: {accuracy:.4f}')

    def tune(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Re-initialize weights and biases for tuning
        self.initialize_weights_and_biases()
        # Train the model with new hyperparameters
        self.fit()


# Example usage of CustomMLP class:
mlp = CustomMLP(hidden_layers=2, neurons_per_layer=64,
                activation_function='relu', learning_rate=0.001, epochs=100)
mlp.fit()
mlp.test()
