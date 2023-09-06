import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
import os

# learning_rate = 0.001
# num_epochs = 800
# batch_size = 64

class RNNModel:

    def __init__(self, input_size=300, hidden_size=64, output_size=1, num_layers=1, model_name=None, verbose_level=0):
        class RNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_layers):
                super(RNN, self).__init__()
                self.hidden_size = hidden_size
                self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                   batch_first=True, num_layers=num_layers, dropout=0.2)
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

    def getOptimizer(self, learning_rate):
        return optim.Adam(self.model.parameters(),lr=learning_rate)
    def train(self):
        return self.model.train()
    def eval(self):
        return self.model.eval()
    def getOutput(self, inputs):
        return self.model(inputs)
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"INFO: RNN model saved at the path: {path}")
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

    def train_RNN(self, X_train, y_train, num_epochs, batch_size, learning_rate, X_eval=None, y_eval=None,
                  eval_decision_include=False):
        # defining Hyperparameter
        criterion = self.criterion

        optimizer = optim.Adam(self.model.parameters(),lr=learning_rate)

        patience, patience_counter = 100, 0
        best_avg_loss = 1
        best_eval_loss = 100
        # Training loop
        self.model = self.model.double()

        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()
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

            # self.model.eval()
            # with torch.no_grad():
            #     X_eval = torch.stack(X_eval).to(self.device)
            #     outputs_eval = self.model(X_eval)
            #     self.debug(1, f"X_test size is: {X_eval.size()} and outputs size is: {outputs.size()}")
            #     predicted_labels = torch.round(outputs)
            #     criterion = self.criterion
            #     self.debug(1, f" y_eval.size is:{y_eval.size()}")
            #     print(f"{type(y_eval)} and {y_eval.size()}")
            #     # y_test = torch.as_tensor(y_test, dtype=torch.float).clone().detach().unsqueeze(1).to(self.device)
            #     y_eval = self.convertTensor1DTo2D(y_eval)
            #     print(f"{type(y_eval)} and {y_eval.size()}")
            #     eval_loss = criterion(outputs_eval, y_eval)
            #     eval_accuracy = (predicted_labels == y_eval).sum().item() / len(y_test)
            if not (X_eval is None) :
                y_pred = self.test_RNN(X_eval, y_eval)
                y_eval_tensor = self.convertTensor1DTo2D(y_eval)
                # print(f"here: y_pred: {y_pred} and y_eval:{y_eval_tensor}")
                eval_loss = criterion(torch.tensor(y_pred), y_eval_tensor)
                # print(f"Info: Epoch {epoch} eval_loss: {eval_loss}")
            else:
                eval_loss = 0

            if eval_decision_include:
                cond_eval = eval_loss <= best_eval_loss
            else:
                cond_eval = True

            if avg_loss < best_avg_loss and cond_eval:
                best_avg_loss = avg_loss
                best_eval_loss = eval_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Path("../models/", self.model_name))
                print(f'Info: Epoch {epoch}, train_loss {avg_loss}, eval_loss {eval_loss}, '
                      f'best Model saved with the loss {best_avg_loss}')

            else:
                patience_counter +=1
                if patience_counter >= patience:
                    print(f'Early stopping on epoch {epoch}')
                    break

            print(f'Info: Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Eval_loss: {eval_loss:.4f}')

    def test_RNN(self, X_test, y_test=None):
    # Evaluation
        self.model = self.model.double()

        self.model.eval()
        with torch.no_grad():
            # X_test_padded = pad_sequences(X_test)
            # X_test_padded = torch.stack(X_test_padded).to(device)
            X_test = torch.stack(X_test).to(self.device)
            # y_test = torch.tensor(y_test, dtype=torch.float).unsqueeze(1).to(device)
            outputs = self.model(X_test)
            # self.debug(3, f"X_test size is: {X_test.size()} and outputs size is: {outputs.size()}")
            predicted_labels = torch.round(outputs)

            if y_test is not None:
                # defining Hyperparameter
                criterion = self.criterion

                self.debug(1, f" y_test.size is:{y_test.size()}")
                # print(f"{type(y_test)} and {y_test.size()}")
                # y_test = torch.as_tensor(y_test, dtype=torch.float).clone().detach().unsqueeze(1).to(self.device)
                y_test = self.convertTensor1DTo2D(y_test)
                # print(f"{type(y_test)} and {y_test.size()}")
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
        return torch.as_tensor(y, dtype=torch.float64).clone().detach().unsqueeze(1).to(self.device)

    def checkAccuracy(self, predicted, actual):
        tp, tn, fp, fn = 0, 0, 0, 0
        # print(f"predicted : {predicted}")
        y_pred = predicted.flatten()
        y_true = actual.flatten()
        for true_label, pred_label in zip(y_true, y_pred):
            if pred_label == true_label:
                if true_label:
                    tp += 1
                else:
                    tn += 1
            elif pred_label == 1 and true_label == 0:
                fp += 1
            elif pred_label == 0 and true_label == 1:
                fn += 1


        self.debug(0, f"Info: True positive  : {tp}")
        self.debug(0, f"Info: True negative  : {tn}")
        self.debug(0, f"Info: False positive : {fp}")
        self.debug(0, f"Info: False negative : {fn}")

        return tp, tn

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