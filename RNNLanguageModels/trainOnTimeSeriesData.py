
import torch
import argparse
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from timeSeriesDataSet import TimeSeriesDataSet
from rnnModel import RNNModel


data_dir = '../data/'
model_dir = '../models/'
verbose = 1

# rnn hyperparameters
seq_length = 512
num_epochs = 300
batch_size = 32
learning_rate = 0.001

# rnn model params
input_size  = 4
hidden_size = 256
output_size = 1
num_layers  = 2
model_name  = "modelRNN_timeSeries.pt"

def debug(verbose_level:int, str:str):
    if verbose >= verbose_level:
        print(str)
def read_items_from_file(file_path):
    items = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace and newline characters
            items.append(line)
    return items

def train_model(model:RNNModel, num_epochs:int, trainloader:DataLoader, valloader:DataLoader,
                len_trainset:int, len_valset:int):
    print(f"Start model training")

    best_acc = 0
    patience, patience_counter = 100, 0

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


def test_model(model:RNNModel, testloader:DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        for i, batch_data in enumerate(testloader):
            print(f"batch_data: {batch_data}")
            X_batch = batch_data['data'].to(device)
            y_batch = batch_data['label'].to(device).long()
            y_batch = y_batch.float().unsqueeze(1)

            # y_pred = model(X_batch)
            y_pred = model.getOutput(X_batch)

            # class_predictions = nn.functional.log_softmax(y_pred, dim=1).argmax(dim=1)
            class_predictions = torch.tensor([0 if x < 0.5 else 1 for x in torch.flatten(y_pred)])
            # class_predictions = torch.flatten(y_pred)
            # print(f"class_predictions: {class_predictions}")
    print(f"class_predictions is {class_predictions}")
    print(f"y_batch: {y_batch.flatten()}")
    # if len(class_predictions.flatten()) > 1:
    debug (0, f"check the Query...since class prediction size is: {len(class_predictions.flatten())}")
    return class_predictions.flatten()


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
        file_list = read_items_from_file(args.file_name)
        debug(2, f"The file names to collect data from are: {file_list}")
        timeSeriesData = TimeSeriesDataSet(file_list=file_list, target_col='label', seq_length=seq_length)
        trainloader, testloader = timeSeriesData.get_loaders(batch_size=batch_size)

        train_model(model=rnn_model, num_epochs=num_epochs, trainloader=trainloader, valloader=testloader,
                    len_trainset=timeSeriesData.get_train_length(),
                    len_valset=timeSeriesData.get_test_length())

