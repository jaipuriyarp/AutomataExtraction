import random
import torch
import numpy as np



from transformerModel import Transformer, load_model, save_model

verbose = 0
maxlength = 24
## depth = maxlength/2 : (imp note) mentioned below in main function

# Define hyperparameters
input_size = 2
hidden_size = 64
output_size = 1
num_layers = 2

# Define hyperparameters
learning_rate = 0.001
num_epochs = 500
batch_size = 32
model_name = "modelRNN_lang8_balancedParenthesis.pt"

def debug(verbose_level, str):
    if verbose >= verbose_level:
        print(str)

def generate_balanced_parentheses(n):
    if n == 0:
        return ['']

    results = []
    for i in range(n):
        for left in generate_balanced_parentheses(i):
            for right in generate_balanced_parentheses(n - i - 1):
                results.append(f'({left}){right}')

    return results

def generate_balanced_parentheses_up_to_depth(k):
    print(f"INFO: Starting to generate +ve examples..")
    all_parentheses = []
    for i in range(k + 1):
        all_parentheses.extend(generate_balanced_parentheses(i))
    print(f"INFO: Done generating +ve examples..")
    return all_parentheses

def generate_unbalanced_parentheses(n):
    if n == 0:
        return ['']

    results = []
    for i in range(n):
        for left in generate_unbalanced_parentheses(i):
            for right in generate_unbalanced_parentheses(n - i - 1):
                results.append(f'){left}({right}')

    return results
def generate_unbalanced_parentheses_up_to_depth(k):
    print(f"INFO: Starting to generate -ve examples..")
    all_parentheses = []
    for i in range(k + 1):
        all_parentheses.extend(generate_unbalanced_parentheses(i))
    print(f"INFO: Done generating -ve examples..")
    return all_parentheses

def generate_unbalanced_parantheses_from_posL(posL):
    print(f"INFO: Starting to generate -ve examples from +ve ones...")
    negL = []
    for x in posL:
        random_index = random.randint(0, len(x))
        if len(x) == maxlength: ## deleting a letter
            y = x[:random_index] + x[random_index+1:]
        else: ## adding a letter
            y = x[:random_index] + random.choice(['(', ')']) + x[random_index:]
        negL.append(y)
    return negL
def generate_one_parantheses_up_to_depth(k):
    print(f"INFO: Starting to generate one parentheses -ve examples..")
    all_parentheses = []
    for i in range(k + 1):
        str1, str2 = "", ""
        if i > 0:
            for _ in range(i):
                str1 += "("
            all_parentheses.append(str1)
            for _ in range(i):
                str2 += ")"
            all_parentheses.append(str2)
    print(f"INFO: Done generating one parentheses -ve examples..")
    return all_parentheses


def one_hot_encoding(letter):
    alphabet = "()"
    encoding = np.zeros((len(alphabet, )), dtype=int)  # Initialize an array of zeros with length 26
    if len(letter) != 1:
        raise Exception("this function returns one-hot encoding of only one letter at a time")
    encoding[alphabet.index(letter)] = 1
    return encoding
def encode_sequence(sequence):
    debug(2, f"sequence: {sequence}")
    x = torch.tensor(np.array([one_hot_encoding(word) for word in sequence]))
    debug(2, f"size of x: {x.size()}")
    target_seq = torch.zeros(maxlength, input_size)
    if x.size()[0] > 0:
        target_seq[0:x.size(0), :] = x
    debug(2, f"size of target: {target_seq.size()}")
    debug(2, f"target: {target_seq}")
    return target_seq

def generateExamples(depth):
    posL = generate_balanced_parentheses_up_to_depth(k=depth)
    negL = generate_one_parantheses_up_to_depth(k=depth)
    debug(0, f"INFO: length of negL at stage 1: {len(negL)}")
    negL += generate_unbalanced_parentheses_up_to_depth(k=depth)
    debug(0, f"INFO: length of negL at stage 2: {len(negL)}")
    negL += generate_unbalanced_parantheses_from_posL(posL)
    debug(0, f"INFO: length of negL at stage 3: {len(negL)}")
    debug(1, f"posL is: {posL}")
    debug(1, f"negL is: {negL}")

    return posL, negL

def create_datasets(depth, train):
    X = []
    y = []
    posL, negL = generateExamples(depth)
    result = posL + negL
    random.shuffle(result)
    debug(3, f"result: {result}")
    for sequence in result:
        X.append(encode_sequence(sequence))
        y.append(1 if sequence in posL else 0)
    debug(3, "X =" + str(X))
    debug(3, "y=" + str(y))
    print(f"Info: Depth of the samples generated: {depth}")
    if train:
        print(f"Info: Number of positive examples in training: {y.count(1)}")
        print(f"Info: Number of negative examples in training: {y.count(0)}")
    else:
        print(f"Info: Number of positive examples in test: {y.count(1)}")
        print(f"Info: Number of negative examples in test: {y.count(0)}")
    return X, torch.tensor(y, dtype=torch.float)

if __name__ == "__main__":


    transformer_model = Transformer(input_dim=input_size, hidden_dim=hidden_size,
                                    num_layers=num_layers, output_dim=output_size)

    # Load the model saved already in the models folder
    transfomerModelPath = "../models/" + model_name
    transformer_model = load_model(transformer_model, transformer_model)
    needTraining = True

    depth = int(maxlength/2)
    if needTraining:
        X_train, y_train = create_datasets(depth, train=True)
        print(f"Info: Length of X(input) for training: {len(X_train)}")
        print(f"Info: Size of y(label) tensor for training: {y_train.size()}")
        train_transfomer(X_train, y_train, num_epochs, batch_size, learning_rate)



    else:
        print(f"Info: Training is skipped!")

    depth=9
    X_test, y_test = create_datasets(depth, train=False)
    print(f"Info: Length of X(input) for testing: {len(X_test)}")
    print(f"Info: Size of y(label) tensor for testing: {y_test.size()}")
    predicted = transformer_model.test_RNN(X_test, y_test)
    y_test = transformer_model.convertTensor1DTo2D(y_test).numpy()
    # print(f"{y_test} and {predicted}")
    pos, neg = transformer_model.checkAccuracy(predicted=predicted, actual=y_test)
    statistics(numSamples, pos, neg)
    transformer_model.getScores(predicted, y_test)