import os
import psutil
import time
import warnings
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec, KeyedVectors

#gensim.__version__ = 4.3.1

verbose = 0
maxlength = 20
UpperBound = 9999999999

# Define hyperparameters
input_size = 1
hidden_size = 64
output_size = 1
num_layers = 2
# Define hyperparameters
learning_rate = 0.0001
num_epochs = 500
batch_size = 32
modelName = "modelRNNOnQ.pt"

# Define the loss function
criterion = nn.BCELoss()

def debug(verbose_level, str):
    if verbose >= verbose_level:
        print(str)

def special_word():
    # change the code for appending any special word for testing RNN
    word = generateNewWord()
    k = random.randint(1, maxlength)
    print(f'special word chosen is : {word} and sentence length is: {k}')
    if k % 2 != 0:
        k += 1
    special_sentence = [word if i % 2 == 0 else "#" for i in range(k)]
    return special_sentence

def genSpecialNegExample(num_examples, lang):
    # TODO: create examples without '#' as well : Already added due to length selection
    wordL = []
    while num_examples > 0:
        if lang == 1 : #L = (a#)^n
            k = random.randint(1, maxlength-2)
            word = random.randint(1,UpperBound)
            x = [word if i % 2 == 0 else '#' for i in range(k)]
            if x[-1] != '#':
                x.append('#')
            word = random.randint(1,UpperBound)
            while word == x[0]:
                word = random.randint(1,UpperBound)
            x.append(word)
            x.append('#')
            k = random.randint(0, maxlength-len(x))
            y = [x[0] if i%2 == 0 else '#' for i in range(k)]
            x = x + y
            wordL.append(x)

        elif lang == 2 : # L = first and last letter equal
            break

        num_examples -= 1
    # print("special Negatives:", wordL)
    return wordL
def generateTypeExamples(num_examples, lang, pos):
    wordL = []
    while num_examples > 0:
        k = random.randint(1, maxlength)

        # L = (a#)^n where a \in Q+ and # is encoded as 0.
        if lang == 1:
            if pos:
                word = random.randint(1,UpperBound)
                if k % 2 != 0: #To add '# at the end of positive examples if k is odd
                    k += 1
                wordL.append([word if i%2 == 0  else '#' for i in range(k)])
            else:
                x = []
                for i in range(k):
                    if i % 2 == 0:
                        word = random.randint(1,UpperBound)
                        x.append(word)
                    else:
                        x.append("#")
                wordL.append(x)

        # L = axa, axya # first and last letter is same
        elif lang == 2:
            k = 2 * k  # will drop the delimiter symbol '#'  while encoding
            if k > 2:
                x = [random.randint(1, UpperBound) if i % 2 == 0 else '#' for i in range(k - 2)]
                # print(f"length of x:{len(x)} and chosen k is {k/2}")
                if pos:
                    x.append(x[0])
                else:
                    p = random.randint(1, UpperBound)
                    while p == x[0]:
                        p = random.randint(1, UpperBound)
                    x.append(p)
            else:
                x = [random.randint(1, UpperBound)]
            x.append('#')
            wordL.append(x)

        num_examples -= 1

    return wordL

def generateExamples(num_examples, lang):
    posL_count = int(num_examples / 2)
    negL_count = num_examples - posL_count
    negL = genSpecialNegExample(int(negL_count / 2), lang)
    negL_count = negL_count - len(negL)
    if posL_count != 0:
        posL = generateTypeExamples(posL_count, lang=lang, pos=True )
    if negL_count != 0:
        negL = negL + generateTypeExamples(negL_count, lang=lang, pos=False)
    debug(2, "positive examples:" + str(posL))
    debug(2, "negative examples:" + str(negL))
    return posL, negL

def encode_sequence(sequence, lang):
    debug(2, "sequence:" + str(sequence))
    # x = torch.tensor(np.array([w2v_model[word] for word in sequence]))
    if lang == 1:
        x = torch.tensor(np.array([[word] if word != '#' else [0]  for word in sequence]))
    elif lang == 2:
        x = torch.tensor(np.array([[word] for word in sequence if word != '#']))
    debug(1, f"size of x: {x.size()}")
    target_seq = torch.zeros(maxlength, 1)
    target_seq[0:x.size(0),:] = x
    debug(1, f"size of target: {target_seq.size()}")
    debug(2, f"target: {target_seq}")
    return target_seq

def create_datasets(num_examples, lang, train=False):
    X = []
    y = []
    print(f"Info: Generating train={train} examples...")
    posL, negL = generateExamples(num_examples, lang)
    result = posL + negL
    random.shuffle(result)
    for sequence in result:
        X.append(encode_sequence(sequence, lang))
        y.append(1 if sequence in posL else 0)
    debug(0, "X =" + str(X))
    debug(0, "y=" + str(y))
    print(f"Info: Total number of samples generated: {num_examples}")
    if train:
        print(f"Info: Number of positive examples in training: {y.count(1)}")
        print(f"Info: Number of negative examples in training: {y.count(0)}")
    else:
        print(f"Info: Number of positive examples in test: {y.count(1)}")
        print(f"Info: Number of negative examples in test: {y.count(0)}")
    print(f"Info: Finished Generating train={train} examples.")
    return X, torch.tensor(y, dtype=torch.float)


def checkAccuracy(predicted, actual):
    pos, neg = 0, 0
    #print(predicted)
    for i,x in enumerate(actual):
        for j,y in enumerate(x):
            if predicted[i,j] == y:
                if y == 1:
                    pos += 1
                else:
                    neg += 1

    print(f"Info: Number of +ve samples identified correctly: {pos}")
    print(f"Info: Number of -ve samples identified correctly: {neg}")


def createRNNModel():
    # Define the RNN model
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
    model = RNN(input_size, hidden_size, output_size, num_layers).to(device)
    return model

def load_RNN_model(model):
    model_path = "./models/" + modelName
    needTraining = False
    if not os.path.exists(model_path):
        # raise Exception (f"No model exists at {model_path}, "
        #                  f"Please try creating a new RNN model or check the path or model name again")
        print(f"Warning: Model doesn't exist at path {model_path}, So training is needded for a new RNN model")
        needTraining = True
    else:
        print(f"Info: RNN Model loaded Successfully!")
        model.load_state_dict(torch.load(model_path))
        print(f'Info: Model {model}')
    return model, needTraining

def train_RNN(model, X_train, y_train, device):
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    patience, patience_counter = 50, 0
    best_avg_loss = 1
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            inputs = torch.stack(X_train[i:i+batch_size]).to(device)
            labels = y_train[i:i+batch_size].unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            # print("inputs:", inputs)
            # print("outputs:",outputs)
            # print("labels:", labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(X_train) / batch_size)

        if avg_loss <= 0.99*best_avg_loss:
            best_avg_loss = avg_loss
            torch.save(model.state_dict(), Path("models/", modelName))
            print(f'Info: Epoch {epoch} best Model saved with the loss {best_avg_loss}')

        else:
            patience_counter +=1
            if patience_counter >= patience:
                print(f'Early stopping on epoch {epoch}')
                break

        print(f'Info: Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

def test_RNN(model, X_test, y_test, device):
# Evaluation
    with torch.no_grad():
        model.eval()
        # X_test_padded = pad_sequences(X_test)
        # X_test_padded = torch.stack(X_test_padded).to(device)
        X_test = torch.stack(X_test).to(device)
        # y_test = torch.tensor(y_test, dtype=torch.float).unsqueeze(1).to(device)
        y_test = torch.as_tensor(y_test, dtype=torch.float).clone().detach().unsqueeze(1).to(device)
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
        predicted_labels = torch.round(outputs)
        accuracy = (predicted_labels == y_test).sum().item() / len(y_test)
        # print("actual_label:", y_test)
        # print("predicted_labels:", predicted_labels)
        # print("=:", predicted_labels==y_test)
        # print("sum:", (predicted_labels==y_test).sum())
        # print("item:", (predicted_labels == y_test).sum().item())
        # print(predicted_labels[-1])
        checkAccuracy(predicted_labels.numpy(),y_test.numpy())
        print(f"Info: Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RNN_model = createRNNModel()
    # Load the model saved already in the models folder
    RNN_model, needTraining = load_RNN_model(RNN_model)

    #select langugage:
    lang = 2
    needTraining = True

    if needTraining:
        X_train, y_train = create_datasets(100000, lang, train=True)
        print(f"Info: Length of X(input) for training: {len(X_train)}")
        print(f"Info: Size of y(label) tensor for training: {y_train.size()}")
        train_RNN(RNN_model, X_train, y_train, device)
    else:
        print(f"Info: Training is skipped!")

    X_test, y_test = create_datasets(1000, lang, train=False)
    print(f"Info: Length of X(input) for testing: {len(X_test)}")
    print(f"Info: Size of y(label) tensor for testing: {y_test.size()}")
    test_RNN(RNN_model, X_test, y_test, device)
