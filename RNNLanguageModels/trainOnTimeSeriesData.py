import math

import torch
import argparse
import pandas as pd
import torch.nn as nn
from pathlib import Path
from timeSeriesDataSet import TimeSeriesDataSet
from rnnModel import RNNModel


data_dir = '../data/'
model_dir = '../models/'

# rnn hyperparameters
num_epochs = 100
batch_size = 32
learning_rate = 0.001

# rnn model params
input_size  = 4
hidden_size = 256
output_size = 1
num_layers  = 2
model_name  = "modelRNN_timeSeries.pt"

def load_data(file_name):
    data = pd.read_csv(Path(data_dir, file_name))
    return data

def train_model(model, num_epochs, trainloader, valloader, len_trainset, len_valset):
    print(f"Start model training")

    best_acc = 0
    patience, patience_counter = 200, 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters, lr=learning_rate)
    optimizer = model.getOptimizer(learning_rate)
    for epoch in range(num_epochs):

        # initialize losses
        loss_train_total = 0
        loss_val_total = 0

        # Training loop
        correct_train, total_train = 0, 0
        for i, batch_data in enumerate(trainloader):
            model.train()
            X_batch = batch_data['data'].to(device)
            # print(f"X_batch: {X_batch.size()}")
            y_batch = batch_data['label'].to(device).long()
            # converting y_batch of tensor size (32,) to tensor size (32,1) of float type datatype
            y_batch = y_batch.float().unsqueeze(1)
            optimizer.zero_grad()

            y_pred = model.getOutput(X_batch)
            # print(f"y_pred size: {y_pred.size()}")
            # print(f"y_pred: {y_pred}")
            # print(f"y_batch: {y_batch}")
            # y_pred = torch.heaviside(y_pred, torch.tensor([1.0]))#, dtype=torch.float32))
            # check_pred = [x for x in torch.flatten(y_pred) if x<0 or x is math.nan ]
            # check_batch = [x for x in torch.flatten(y_batch) if x<0 or x is math.nan]
            # print(f"check_pred: {check_pred} and check_batch: {check_batch}")
            loss = criterion(y_pred, y_batch)
            #print("loss", loss)
            loss_train_total += loss.cpu().detach().item() * batch_size

            loss.backward()
            optimizer.step()

            class_predictions_train = [0 if x<0.5 else 1 for x in torch.flatten(y_pred)]
            total_train += y_batch.size(0)
            correct_train += [x == y for x,y in zip(class_predictions_train,torch.flatten(y_batch))].count(True)

        train_acc = correct_train/total_train

        loss_train_total = loss_train_total / len_trainset

        # Validation loop
        with torch.no_grad():
            for i, batch_data in enumerate(valloader):
                model.eval()
                X_batch = batch_data['data'].to(device)
                y_batch = batch_data['label'].to(device).long()
                y_batch = y_batch.float().unsqueeze(1)

                # y_pred = model(X_batch)
                y_pred = model.getOutput(X_batch)
                loss = criterion(y_pred, y_batch)
                loss_val_total += loss.cpu().detach().item() * batch_size

        loss_val_total = loss_val_total / len_valset

        # Validation Accuracy
        correct, total = 0, 0
        with torch.no_grad():
            model.eval()
            for i, batch_data in enumerate(valloader):
                X_batch = batch_data['data'].to(device)
                y_batch = batch_data['label'].to(device).long()
                # y_batch = y_batch.float().unsqueeze(1)

                # y_pred = model(X_batch)
                y_pred = model.getOutput(X_batch)

                # class_predictions = nn.functional.log_softmax(y_pred, dim=1).argmax(dim=1)
                class_predictions = torch.tensor([0 if x<0.5 else 1 for x in torch.flatten(y_pred)])
                # class_predictions = torch.flatten(y_pred)
                # print(f"class_predictions: {class_predictions}")
                # print(f"y_batch: {y_batch}")
                total += y_batch.size(0)
                correct += (class_predictions == y_batch).sum().item()
                # print(f"correct: {correct}")

        acc = correct / total

        # Logging
        # if epoch % 5 == 0:
        #     print(
        #         f'Epoch: {epoch:3d}. Train Loss: {loss_train_total:.4f}. Train Acc.:{train_acc:2.2%} '
        #         f'Val Loss: {loss_val_total:.4f} Acc.: {acc:2.2%}')
        print(
            f'Epoch: {epoch:3d}. Train Loss: {loss_train_total:.4f}. Train Acc.:{train_acc:2.2%} '
            f'Val Loss: {loss_val_total:.4f} Acc.: {acc:2.2%}')

        if acc > best_acc:
            patience_counter = 0
            best_acc = acc
            # torch.save(model.state_dict(), Path(model_dir, model_name))
            model.save(Path(model_dir, model_name))
            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping on epoch {epoch}')
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default=None)
    args = parser.parse_args()

    rnn_model = RNNModel(input_size=input_size, hidden_size=hidden_size,
                         output_size=output_size, num_layers=num_layers, model_name=model_name)

    needTraining = rnn_model.load_RNN_model(Path(model_dir, model_name))

    if needTraining:
        if args.file_name is None:
            raise Exception("Need file name to prepare data.")
        df = load_data(args.file_name)
        timeSeriesData = TimeSeriesDataSet(data=df, target_col='label', seq_length=128)
        trainloader, testloader = timeSeriesData.get_loaders(batch_size=batch_size)

        train_model(model=rnn_model, num_epochs=num_epochs, trainloader=trainloader, valloader=testloader,
                    len_trainset=timeSeriesData.get_train_length(),
                    len_valset=timeSeriesData.get_test_length())
