import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
import os

learning_rate = 0.001
num_epochs = 800
batch_size = 32

class RNNModel:

    def __init__(self, input_size=300, hidden_size=64, output_size=1, num_layers=1, model_name=None, verbose_level=0):
        class RNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_layers):
                super(RNN, self).__init__()
                self.hidden_size = hidden_size
                self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                   batch_first=True, num_layers=num_layers)
                self.fc = nn.Linear(hidden_size, output_size)
                self.sigmoid = nn.Sigmoid()

            def forward(self, input):
                hidden_features, (h_n, c_n) = self.rnn(input)  # (h_0, c_0) default to zeros
                hidden_features = hidden_features[:, -1, :]  # index only the features produced by the last LSTM cell
                out = self.fc(hidden_features)
                out = self.sigmoid(out)
                return out


        # Create the RNN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set the device for training
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        if model_name is None:
            self.model_name = "newLSTMModel.pt"
        else:
            self.model_name = model_name
        self.verbose_level = verbose_level

        # Define the optimizer
        self.criterion = nn.BCELoss()

        self.model = RNN(input_size, hidden_size, output_size, num_layers).to(self.device)
        print(f'Info: RNN Model instantiation is:  {self.model}')

    def debug(self, verbose, str):
        if self.verbose_level >= verbose:
            print(str)
    def load_RNN_model(self, model_path):
        needTraining = False
        if not os.path.exists(model_path):
            print(f"Warning: Model doesn't exist at path {model_path}, So training is needed for a new RNN model")
            needTraining = True
        else:
            print(f"Info: The model exists at {os.path.exists(model_path)}")
            self.model.load_state_dict(torch.load(model_path))
            print(f"Info: RNN Model loaded Successfully!")
            # print(f'Info: Model {self.model}')
        return needTraining

    def train_RNN(self, X_train, y_train, num_epochs, batch_size, learning_rate):
        # defining Hyperparameter
        criterion = self.criterion

        optimizer = optim.Adam(self.model.parameters(),lr=learning_rate)

        patience, patience_counter = 200, 0
        best_avg_loss = 1
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                inputs = torch.stack(X_train[i:i+batch_size]).to(self.device)
                labels = y_train[i:i+batch_size].unsqueeze(1).to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                # print("inputs:", inputs)
                # print("outputs:",outputs)
                # print("labels:", labels)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (len(X_train) / batch_size)

            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Path("../models/", self.model_name))
                print(f'Info: Epoch {epoch} best Model saved with the loss {best_avg_loss}')

            else:
                patience_counter +=1
                if patience_counter >= patience:
                    print(f'Early stopping on epoch {epoch}')
                    break

            print(f'Info: Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    def test_RNN(self, X_test, y_test=None):
    # Evaluation
        with torch.no_grad():
            self.model.eval()
            # X_test_padded = pad_sequences(X_test)
            # X_test_padded = torch.stack(X_test_padded).to(device)
            X_test = torch.stack(X_test).to(self.device)
            # y_test = torch.tensor(y_test, dtype=torch.float).unsqueeze(1).to(device)
            outputs = self.model(X_test)
            self.debug(1, f"X_test size is: {X_test.size()} and outputs size is: {outputs.size()}")
            predicted_labels = torch.round(outputs)

            if y_test is not None:
                # defining Hyperparameter
                criterion = self.criterion

                self.debug(1, f" y_test.size is:{y_test.size()}")
                print(f"{type(y_test)} and {y_test.size()}")
                # y_test = torch.as_tensor(y_test, dtype=torch.float).clone().detach().unsqueeze(1).to(self.device)
                y_test = self.convertTensor1DTo2D(y_test)
                print(f"{type(y_test)} and {y_test.size()}")
                loss = criterion(outputs, y_test)
                accuracy = (predicted_labels == y_test).sum().item() / len(y_test)
                print(f"Info: Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")
                # self.checkAccuracy(predicted_labels.numpy(), y_test.numpy())
            # print("actual_label:", y_test)
            # print("predicted_labels:", predicted_labels)
            # print("=:", predicted_labels==y_test)
            # print("sum:", (predicted_labels==y_test).sum())
            # print("item:", (predicted_labels == y_test).sum().item())
            # print(predicted_labels[-1])
        return predicted_labels.numpy()

    def convertTensor1DTo2D(self, y):
        return torch.as_tensor(y, dtype=torch.float).clone().detach().unsqueeze(1).to(self.device)

    def checkAccuracy(self, predicted, actual):
        pos, neg = 0, 0
        # print(f"predicted : {predicted}")
        for i, x in enumerate(actual):
            for j, y in enumerate(x):
                if predicted[i, j] == y:
                    if y == 1:
                        pos += 1
                    else:
                        neg += 1

        self.debug(1, f"Info: Total number of +ve data points identified correctly : {pos}")
        self.debug(1, f"Info: Total number of -ve data points identified correctly : {neg}")

        return pos, neg

    def getScores(self, predicted, actual):
        if not (isinstance(predicted, np.ndarray) and isinstance(actual, np.ndarray)):
            raise Exception(f"Please ensure that the type of predicted: {type(predicted)} "
                            f"and ground truth:{type(actual)} are of np.ndarray type")
        predicted_flat = predicted.flatten()
        actual_flat = actual.flatten()

        # Calculate accuracy
        accuracy = np.mean(predicted_flat == actual_flat)

        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(actual_flat, predicted_flat,
                                                                   average='weighted')

        # Print the evaluation metrics
        print("Result: Accuracy :", accuracy)
        print("Result: Precision:", precision) # Higher precision means fewer false positives.
        print("Result: Recall   :", recall) # Higher recall means fewer false negatives. #imp to not miss true positive
        print("Result: F1 Score :", f1)