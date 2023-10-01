# models/mlp_updated.py

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

class CustomMLP:
    def __init__(self, hidden_layers, neurons_per_layer, activation_function, learning_rate, epochs):
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_size = None  # Initialize input_size
        self.output_size = None  # Initialize output_size

        self.data, self.labels = self.load_and_preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=42)

        self.model = self.build_model()

    def load_and_preprocess_data(self):
        data = []  # List to store image data
        labels = []  # List to store labels

        # Define a function to extract the expression label from the file name
        def extract_expression_label(file_name):
            return file_name.split('_')[1]  # Assuming the expression label is between '_' and the file extension

        # Load data and labels
        dataset_dir = 'C:/Users/admin/Desktop/Facial-Recognition-System/data/yalefaces_resized'  # Update with your dataset directory
        for file_name in os.listdir(dataset_dir):
            img_path = os.path.join(dataset_dir, file_name)
            if img_path.lower().endswith('.png'):
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = np.array(img)  # Convert image to numpy array
                img_array = img_array.astype('float32') / 255.0  # Normalize pixel values to range [0, 1]
                data.append(img_array.flatten())  # Flatten image into a 1D array
                labels.append(extract_expression_label(file_name))

        # Convert string labels to numerical labels using LabelEncoder
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        data = np.array(data)
        labels = np.array(labels)
        self.input_size = data.shape[1]
        self.output_size = len(np.unique(labels))
        return data, labels

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(self.input_size,)))  # Input size from data
        for _ in range(self.hidden_layers):
            model.add(tf.keras.layers.Dense(self.neurons_per_layer, activation=self.activation_function))
        model.add(tf.keras.layers.Dense(self.output_size, activation='softmax'))  # Output size from data

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self):
        # Implement early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(self.X_train, self.y_train, epochs=self.epochs, validation_split=0.2, verbose=2, callbacks=[early_stopping])
        
        # Calculate metrics on the test set
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, np.argmax(y_pred, axis=1))
        # Set zero_division parameter to control the behavior
        precision = precision_score(self.y_test, np.argmax(y_pred, axis=1), average='macro', zero_division=0)
        recall = recall_score(self.y_test, np.argmax(y_pred, axis=1), average='macro', zero_division=0)
        f1 = f1_score(self.y_test, np.argmax(y_pred, axis=1), average='macro', zero_division=0)

        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        
        return history

    def save_model_and_data(self):
        # Save the trained model
        joblib.dump(self.model, 'face_recognizer.pkl')

        # Save face embeddings and labels
        np.save('face_embeddings.npy', self.data)
        np.save('face_labels.npy', self.labels)

# Example usage of CustomMLP class:
mlp = CustomMLP(hidden_layers=2, neurons_per_layer=64, activation_function='relu', learning_rate=0.001, epochs=100)
history = mlp.fit()

# Save the model and data after training
mlp.save_model_and_data()