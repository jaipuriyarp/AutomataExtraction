import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define the alphabet and create a mapping of characters to indices
alphabet = 'abcdefghijklmnopqrstuvwxyz'
char_to_idx = {char: idx for idx, char in enumerate(alphabet)}

# Function to encode a word into a one-hot representation
def encode_word(word):
    # encoded = torch.zeros(len(word), len(alphabet))
    encoded = torch.zeros(10, len(alphabet))
    for i, char in enumerate(word):
        encoded[i, char_to_idx[char]] = 1
    # print("word:" + str(word) + "encoded:" + str(encoded) )
    return encoded

# Function to generate training examples
def generate_SpecialExamples(num_examples, pos=True):
    maxlength = 10
    wordL = []
    while num_examples > 0:
        if pos:
            word = ''.join(random.choices(alphabet, k=random.randint(2, maxlength-2)))
            word = word + word[-1]
            if len(word) < 10:
                succ = ''.join(random.choices(alphabet, k=random.randint(0, 10 - len(word))))
                word = word + succ
            wordL.append(word)
            num_examples -= 1
        else:
            posFound = False
            word = ''.join(random.choices(alphabet, k=random.randint(2, maxlength-2)))
            for i in range(len(word) - 1):
                if word[i] == word[i + 1]:
                    posFound = True
            if not(posFound):
                wordL.append(word)
                num_examples -= 1
    return wordL

def generate_typeExamples(num_examples):
        maxLength = 10
        posL, negL = [], []
        posL_count = int(num_examples/2)
        negL_count = num_examples - posL_count
        while not(posL_count == 0 or negL_count == 0):
            posFound = False
            word = ''.join(random.choices(alphabet, k=random.randint(2, maxLength)))
            for i in range(len(word) - 1):
                if word[i] == word[i + 1]:
                    posFound = True
                    posL.append(word)
                    posL_count -= 1
            if not(posFound):
                negL.append(word)
                negL_count -= 1
        if posL_count != 0:
            posL = posL + generate_SpecialExamples(posL_count)
        if negL_count !=0:
            negL = negL + generate_SpecialExamples(negL_count, False)
        return posL, negL
def generate_examples(num_examples,train=False):
    X = []
    y = []
    # for _ in range(num_examples):
    #     word = np.random.choice(list(alphabet), np.random.randint(2, 10))
    posL, negL = generate_typeExamples(num_examples)
    result = np.random.choice(np.concatenate([posL, negL]), num_examples, replace=False)
    for word in result:
        X.append(encode_word(word))
        #y.append(1 if any(word[i] == word[i+1] for i in range(len(word)-1)) else 0)
        y.append(1 if word in posL else 0)
        #print("word:" + str(word))
    #print("y:" + str(y))
    print("Total number of samples generated:", num_examples)
    if train:
        print("Number or positive examples in training:", y.count(1))
        print("Number or negative examples in training:", y.count(0))
    else:
        print("Number or positive examples in test:", y.count(1))
        print("Number or negative examples in test:", y.count(0))
    return X, torch.tensor(y, dtype=torch.float)

# Generate training examples
#X_train, y_train = generate_examples(10000)
#X_test, y_test = generate_examples(1000)

X_train, y_train = generate_examples(100000, train=True)
X_test, y_test = generate_examples(1000)
print("size of each input x" + str(X_train[0].size()))
#print(y_test.size())

#print(X_test, y_test)

# Pad sequences to ensure equal length
# def pad_sequences(sequences):
#     max_length = max(len(seq) for seq in sequences)
#     padded_seqs = []
#     for seq in sequences:
#         if len(seq) < max_length:
#             padding = torch.zeros(max_length - len(seq), len(alphabet))
#             padded_seq = torch.cat((seq, padding))
#             padded_seqs.append(padded_seq)
#         else:
#             padded_seqs.append(seq)
#     return padded_seqs
#
# X_train = pad_sequences(X_train)
# X_test = pad_sequences(X_test)

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

    print("Number of +ve samples identified correctly:", pos)
    print("Number of -ve samples identified correctly:", neg)




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
        # hidden = self.init_hidden(input.size(0))
        # output, _ = self.rnn(input, hidden)
        # output = self.fc(output[:, -1, :])
        # output = self.sigmoid(output)
        # return output
        hidden_features, (h_n, c_n) = self.rnn(input)  # (h_0, c_0) default to zeros
        hidden_features = hidden_features[:, -1, :]  # index only the features produced by the last LSTM cell
        out = self.fc(hidden_features)
        out = self.sigmoid(out)
        return out
    # def forward(self, input):
    #     _, (hidden, _) = self.rnn(input)
    #     output = self.fc(hidden[-1])
    #     output = self.sigmoid(output)
    #     return output

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
input_size = len(alphabet)
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 15
batch_size = 32
num_layers = 1

# Create the RNN model
model = RNN(input_size, hidden_size, output_size, num_layers).to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Evaluation
with torch.no_grad():
    model.eval()
    # X_test_padded = pad_sequences(X_test)
    # X_test_padded = torch.stack(X_test_padded).to(device)
    X_test = torch.stack(X_test).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float).unsqueeze(1).to(device)
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
    print(f"Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")