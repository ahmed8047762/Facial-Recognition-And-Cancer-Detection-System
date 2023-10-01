import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

class SkinCancerClassifier:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.X = self.data.drop(columns=['label'])
        self.y = self.data['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def show_samples(self, N, train=True):
        samples = self.X_train if train else self.X_test
        print(samples.head(N))

    def build_mlp(self, hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam'):
        self.clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, activation=activation, solver=solver)

    def train(self):
        self.clf.fit(self.X_train, self.y_train)

    def display_training_error(self):
        train_accuracy = self.clf.score(self.X_train, self.y_train)
        print(f'Training Accuracy: {train_accuracy:.2f}')

    def evaluate_test_data(self):
        predictions = self.clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        confusion = confusion_matrix(self.y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, predictions, average='weighted')

        print(f'Test Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')
        print('Confusion Matrix:')
        print(confusion)

# Example Usage
if __name__ == '__main__':
    data_file = 'C:/Users/admin/Desktop/hmnist_28_28_RGB.csv'
    classifier = SkinCancerClassifier(data_file)

    # Show 5 training samples
    print('Training Samples:')
    classifier.show_samples(5)

    # Build and train MLP model
    classifier.build_mlp(hidden_layer_sizes=(100, 50))
    classifier.train()

    # Display training error
    print('\nTraining Error:')
    classifier.display_training_error()

    # Evaluate on test data and display metrics
    print('\nEvaluation on Test Data:')
    classifier.evaluate_test_data()
