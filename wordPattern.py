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

verbose = 0
maxlength = 20

# Define hyperparameters
input_size = 300
hidden_size = 64
output_size = 1
num_layers = 2
# Define hyperparameters
learning_rate = 0.001
num_epochs = 100
batch_size = 32

# Define the loss function
criterion = nn.BCELoss()

def debug(verbose_level, str):
    if verbose >= verbose_level:
        print(str)

def load_model():
    word2vecPath = "./models/GoogleNews-vectors-negative300.bin"
    warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
    if not os.path.exists(word2vecPath):
        raise Exception("Error: word2vec model doesn't exist at the path:" + str(word2vecPath))
    debug(0,f"Info: word2vec Model at {word2vecPath}")

    if (verbose==2):
        process = psutil.Process(os.getpid())
        pre = process.memory_info().rss
        print("Memory used in GB before Loading the Model: %0.2f" % float(pre / (10 ** 9)))  # Check memory usage before loading the model
        print('-' * 10)

        start_time = time.time()  # Start the timer
        # mem= psutil.virtual_memory()
        # ttl = mem.total  # Toal memory available

    w2v_model = KeyedVectors.load_word2vec_format(word2vecPath, binary=True)#, limit=20000)  # load the model
    # w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True)
    if (verbose==2):
        print("%0.2f seconds taken to load" % float(time.time() - start_time))  # Calculate the total time elapsed since starting the timer
        print('-' * 10)

        print('Finished loading Word2Vec')
        print('-' * 10)

        post = process.memory_info().rss
        print("Memory used in GB after Loading the Model: {:.2f}".format(
            float(post / (10 ** 9))))  # Calculate the memory used after loading the model
        print('-' * 10)

        print("Percentage increase in memory usage: {:.2f}% ".format(
            float((post / pre) * 100)))  # Percentage increase in memory after loading the model
        print('-' * 10)

    debug(0,f"Number of words in vocabs: {len(w2v_model)}")  # Number of words in the vocabulary
    return w2v_model
def test_model(w2v_model):
    # word = w2v_model[100]
    word = random.choices(w2v_model.index_to_key)
    # array_for_hash = w2v_model.get_vector("#")
    # print(array_for_hash)
    if not(word):
        raise Exception("w2v Model not loaded properly!")
    debug (0, "w2v Model loading tested Successfully!")

def genSpecialNegExample(num_examples):
    # TODO: create examples without '#' as well : Already added due to length selection
    wordL = []
    while num_examples > 0:
        k = random.randint(1, maxlength-2)
        word = random.choices(w2v_model.index_to_key)
        x = [word[0] if i % 2 == 0 else '#' for i in range(k)]
        if x[-1] != '#':
            x.append('#')
        word = random.choices(w2v_model.index_to_key)
        while word == x[0]:
            word = random.choices(w2v_model.index_to_key)
        x.append(word[0])
        x.append('#')
        k = random.randint(0, maxlength-len(x))
        y = [x[0] if i%2 ==0 else '#' for i in range(k)]
        x = x + y
        wordL.append(x)
        num_examples -= 1
    # print("special Negatives:", wordL)
    return wordL
def generateTypeExamples(num_examples, pos):
    wordL = []
    while num_examples > 0:
        k = random.randint(1, maxlength)
        if pos:
            word = random.choices(w2v_model.index_to_key)
            if k%2 != 0: #To add '# at the end of positive examples if k is -ve
                k += 1
            wordL.append([word[0] if i%2 == 0  else '#' for i in range(k)])
        else:
            x = []
            for i in range(k):
                if i%2 == 0:
                    word = random.choices(w2v_model.index_to_key)
                    x.append(word[0])
                else:
                    x.append("#")
            wordL.append(x)
        num_examples -= 1
    return wordL

def generateExamples(num_examples):
    posL_count = int(num_examples / 2)
    negL_count = num_examples - posL_count
    negL = genSpecialNegExample(int(negL_count / 2))
    negL_count = negL_count - int(negL_count / 2)
    if posL_count != 0:
        posL = generateTypeExamples(posL_count, pos=True)
    if negL_count != 0:
        negL = negL + generateTypeExamples(negL_count, pos=False)
    debug(2, "positive examples:" + str(posL))
    debug(2, "negative examples:" + str(negL))
    return posL, negL

def encode_sequence(sequence, w2v_model):
    debug(1, "sequence:" + str(sequence))
    x = torch.tensor(np.array([w2v_model[word] for word in sequence]))
    debug(1, "size of x:" + str(x.size()))
    target_seq = torch.zeros(maxlength,300)
    target_seq[0:x.size(0),:] = x
    debug(1, "size of target:" + str(target_seq.size()))
    debug(2, "target:" + str(target_seq))
    return target_seq

def special_word():
    #change the code for appending any special word for testing RNN
    k = random.randint(1, maxlength)
    if k%2 != 0:
        k += 1
    special_sentence = ["Jaipuriyar" if i%2 == 0 else "#" for i in range(k)]
    return special_sentence

def create_datasets(num_examples, w2v_model, train=False):
    X = []
    y = []
    posL, negL = generateExamples(num_examples)
    result = posL + negL
    random.shuffle(result)
    for sequence in result:
        X.append(encode_sequence(sequence, w2v_model))
        y.append(1 if sequence in posL else 0)
    debug(2, "X =" + str(X))
    debug(2, "y=" + str(y))
    print(f"Info: Total number of samples generated: {num_examples}")
    if train:
        print(f"Info: Number of positive examples in training: {y.count(1)}")
        print(f"Info: Number of negative examples in training: {y.count(0)}")
    else:
        #X.append(encode_sequence(special_word(), w2v_model))
        #y.append(1)
        print(f"Info: Number of positive examples in test: {y.count(1)}")
        print(f"Info: Number of negative examples in test: {y.count(0)}")
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
    model_path = "./models/modelRNN.pt"
    needTraining = False
    if not os.path.exists(model_path):
        # raise Exception (f"No model exists at {model_path}, "
        #                  f"Please try creating a new RNN model or check the path or model name again")
        printf(f"Warning: Model doesn't exist at path {model_path}, So training is needded for a new RNN model")
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

        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            torch.save(model.state_dict(), Path("models/", "modelRNN.pt"))
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
        checkAccuracy(predicted_labels.numpy(),y_test.numpy())
        print(f"Info: Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    w2v_model = load_model()
    test_model(w2v_model)
    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RNN_model = createRNNModel()
    # Load the model saved already in the models folder
    RNN_model, needTraining = load_RNN_model(RNN_model)

    if needTraining:
        X_train, y_train = create_datasets(10000, w2v_model=w2v_model, train=True)
        print(f"Info: Length of X(input) for training: {len(X_train)}")
        print(f"Info: Size of y(label) tensor for training: {y_train.size()}")
        train_RNN(RNN_model, X_train, y_train, device)
    else:
        print(f"Info: Training is skipped!")

    X_test, y_test = create_datasets(1000, w2v_model=w2v_model, train=False)
    print(f"Info: Length of X(input) for testing: {len(X_test)}")
    print(f"Info: Size of y(label) tensor for testing: {y_test.size()}")


    test_RNN(RNN_model, X_test, y_test, device)
