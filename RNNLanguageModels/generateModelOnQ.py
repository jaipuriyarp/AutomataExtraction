'''
Usage: This file is used for training and testing the RNN models for languages 1 till 7.
Info: L1 to L7 is the Tomita extended languages (defined already in thesis).
Python version: Python 3.9.0
How to run:
    python <file_name> --lang <language number in integer>
    py generateModelOnQ.py --lang 2
    OR
    python <file_name> --lang <language number in integer> --model_suffix_name <RNN model suffix> \
    --train_file_suffix <file suffix name for training data> --test_file_suffix <file suffix name for test data> --num_examples <number of examples in integer for test data>
    py generateModelOnQ.py --lang 2 --model_suffix_name champa --train_file_suffix champa --test_file_suffix champa --num_examples 50

    Note:
        Second method is recommended if you've any 'file_suffix' option while running the 'dataGeneratorOnQ.py' for generation of train data.
Default value:
    model_suffix_name: None : the value used to identify the RNN model name.
    train_file_suffix: None : the value used to identify the file name for train dataset.
    test_file_suffix: None  : the value used to identify the file name for test dataset.
    num_examples : 100      : the value used for creating test dataset if test file is not found in the data_dir.
    verbose: 0
Note:
    You can change the following values as per requirements, which are not user arguments:
    'maxlength' (present in main function):  the maximum length of the sequences to generate
    'start' and 'maxInteger': the start and end range of elements of sequences respectively.
    'data_dir': the directory name to which the generated dataset is saved to/read from.
    'model_dir': the directory for RNN models to be saved to/load from.
    'verbose' : value for debugging purposes, by default it is set to 0.

Important procedures to follow:
    Please set the same hyperparameters to load an RNN model as set during its training.
    If the training data is not generated already please use the 'dataGeneratorOnQ.py' to generate the training data.
    If the training dataset is saved to a file, which doesn't follow the same name pattern as in 'dataGeneratorOnQ.py'
    or used in the function 'read_file' here, then please change the function as per requirements.
'''

import random
import torch
import argparse
import os
import ast

from rnnModel import RNNModel
from groundTruthFunctions import *
from dataGeneratorOnQ import generateTypeExamples

verbose = 0
maxlength = 20
start = 0
maxInteger = (2^63 -1)
data_dir = "../dataOnQ"
model_dir = "../models"

# Define hyperparameters
input_size = 1
hidden_size = 64
output_size = 1
num_layers = 2

# Define hyperparameters
learning_rate = 0.001
num_epochs = 500
batch_size = 32

def debug(verbose_level: int, string: str):
    if verbose >= verbose_level:
        print(string)

def find_gndFunction(lang: int):
    if lang == 1:
        checkGndLabel = Lang_is_aStar
    elif lang == 2:
        checkGndLabel = Lang_is_abSeq
    elif lang == 3:
        checkGndLabel = Lang_is_abSeq_OddaEvenb
    elif lang == 4:
        checkGndLabel = Lang_is_noTrigrams
    elif lang == 5:
        checkGndLabel = Lang_is_abBothEven
    elif lang == 6:
        checkGndLabel = Lang_is_aMod3b
    elif lang == 7:
        checkGndLabel = Lang_is_aStarbStaraStarbStar
    else:
        raise Exception("No such languages!!")

    return checkGndLabel

def encode_sequence(sequence: list, maxlength: int):
    debug(2, f"sequence: {sequence}")
    x = torch.tensor(np.array([[word] for word in sequence]))
    debug(2, "size of x: {x.size()}")
    target_seq = torch.zeros(maxlength, input_size, dtype=torch.float64)
    if x.size()[0] > 0:
        target_seq[0:x.size(0),:] = x
    debug(2, "size of target: {target_seq.size()}")
    debug(2, "target: {target_seq}")
    return target_seq

def read_file(lang: int, train: bool, pos: bool, file_suffix=None):
    if train:
        if pos:
            file_name = "pos_lang" + str(lang)
        else:
            file_name = "neg_lang" + str(lang)
    else:
        if pos:
            file_name = "pos_lang" + str(lang) + "_test"
        else:
            file_name = "neg_lang" + str(lang) + "_test"

    if file_suffix is not None:
        file_name = file_name + "_" + file_suffix

    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        with open(file_path) as f:
            first_line = f.readline()
            content = f.read()
            list_from_file = ast.literal_eval(content)
        return list_from_file

    debug(0, f"Warning: file not found: {file_path}")
    return []

def seperate_data_for_eval(posL: list, negL: list, _fraction: float):
    random.shuffle(posL)
    random.shuffle(negL)
    posL_eval = posL[:int(len(posL)*_fraction/2)]
    negL_eval = negL[:int(len(negL)*_fraction/2)]
    new_posL = posL[int(len(posL)*_fraction/2):]
    new_negL = negL[int(len(negL)*_fraction/2):]
    return new_posL, new_negL, posL_eval, negL_eval

def find_intersection(list1: list, list2: list):
    # find intersection if any element of list1 present in list2
    list1, list2 = list(list1), list(list2)
    if len(list2) > 0:
        intersection_list = [i for i in list1 if i in list2]
        # intersection_list += [i for i in list2 if i in list1 if i not in intersection_list]
        print(f"intersection_list: {len(intersection_list)}")
        return intersection_list
    return []

def generate_test_data(lang: int, num_examples: int, maxlength: int, train_file_suffix=None, test_file_suffix=None, generate_diff_data_from_train=True):
    posL = read_file(lang, train=False, pos=True, file_suffix=test_file_suffix)
    negL = read_file(lang, train=False, pos=False, file_suffix=test_file_suffix)
    posL_train, negL_train = [], []
    if generate_diff_data_from_train:
        posL_train = read_file(lang, train=True, pos=True, file_suffix=train_file_suffix)
        negL_train = read_file(lang, train=True, pos=False, file_suffix=train_file_suffix)

    posL, negL = set(posL), set(negL)
    while len(posL) < int(num_examples/2):
        posL = posL.union(set(generateTypeExamples(int(num_examples/2), lang, pos=True, maxlength=maxlength)))
        for i in find_intersection(posL, posL_train):
            posL.remove(i)
    file_name = "pos_lang" + str(lang) + "_test"
    if test_file_suffix != "":
        file_name = file_name + "_" + test_file_suffix
    file_path = os.path.join(data_dir,  file_name)
    with open(file_path, "w") as f:
        f.write(f"Total number: {str(len(posL))}\n")
        f.write(str(list(posL)))


    while len(negL) < int(num_examples/2):
        negL = negL.union(set(generateTypeExamples(int(num_examples/2), lang, pos=False, maxlength=maxlength)))
        for i in find_intersection(negL, negL_train):
            negL.remove(i)
    file_name = "neg_lang" + str(lang) + "_test"
    if test_file_suffix != "":
        file_name = file_name + "_" + test_file_suffix
    file_path =  os.path.join(data_dir, file_name)
    with open(file_path, "w") as f:
        f.write(f"Total number: {str(len(negL))}\n")
        f.write(str(list(negL)))

    X, y = [], []
    result = list(posL) + list(negL)
    random.shuffle(result)
    checkGndLabel = find_gndFunction(lang)
    debug(3, f"result: {result}")
    for sequence in result:
        X.append(encode_sequence(sequence, maxlength))
        y.append(1 if checkGndLabel(sequence, False) else 0)

    return X, torch.tensor(y, dtype=torch.float64), y.count(1), y.count(0)
def get_available_datasets(lang: int, validation: bool, train_file_suffix=None):
    X = []
    y = []
    X_eval = []
    y_eval = []
    posL = read_file(lang, train=True ,pos=True, file_suffix=train_file_suffix)
    negL = read_file(lang, train=True ,pos=False, file_suffix=train_file_suffix)
    if posL == [] or negL == [] or posL is None or negL is None:
        raise Exception("Please generate training data first!")

    checkGndLabel = find_gndFunction(lang)

    if validation:
        posL, negL, posL_eval, negL_eval = seperate_data_for_eval(posL, negL, 0.2)

    result = posL + negL
    random.shuffle(result)
    debug(3, f"result: {result}")
    for sequence in result:
        X.append(encode_sequence(sequence, maxlength))
        y.append(1 if checkGndLabel(sequence, False) else 0)
    debug(3, "X =" + str(X))
    debug(3, "y=" + str(y))

    debug(0, f"Info: Number of positive examples in training: {y.count(1)}")
    debug(0, f"Info: Number of negative examples in training: {y.count(0)}")

    if validation:
        result_eval = posL_eval + negL_eval
        random.shuffle(result_eval)
        for sequence in result_eval:
            X_eval.append(encode_sequence(sequence, maxlength))
            y_eval.append(1 if checkGndLabel(sequence, False) else 0)

        debug(0, f"Info: Number of positive examples in eval: {y_eval.count(1)}")
        debug(0, f"Info: Number of negative examples in eval: {y_eval.count(0)}")

    return X, torch.tensor(y, dtype=torch.float64), X_eval, torch.tensor(y_eval, dtype=torch.float64), y.count(1), y.count(0)


def statistics(actual_pos: int, actual_neg: int, predicted_pos: int, predicted_neg: int):
    numSamples = actual_pos + actual_neg

    print(f"Statistics: Accuracy %                                           : "
          f"{(predicted_pos + predicted_neg) / numSamples}")
    print(
        f"Statistics: Inaccuracy %                                         : {(numSamples - (predicted_pos + predicted_neg)) / numSamples}")
    print(f"Statistics: Total number of data points generated                : {numSamples}")
    print(f"Statistics: Total number of data points classified correctly     : {predicted_pos + predicted_neg}")
    print(f"Statistics: Total number of data points misclassified            : {numSamples - (predicted_pos + predicted_neg)}")
    print(f"Statistics: Total number of +ve data points generated            : {actual_pos}")
    print(f"Statistics: Total number of +ve data points identified correctly : {predicted_pos}")
    print(f"Statistics: Total number of -ve data points generated            : {actual_neg}")
    print(f"Statistics: Total number of -ve data points identified correctly : {predicted_neg}")
    print(f"Statistics: Total +ve Accuracy %                                 : {predicted_pos / actual_pos}")
    print(f"Statistics: Total -ve Accuracy %                                 : {predicted_neg / actual_neg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This file is used for the purpose of training and testing RNN models.')
    parser.add_argument('--lang', type=int, help='Specify the language from 1 to 7, for which to train/test an RNN model.')
    parser.add_argument('--model_suffix_name', type=str, default=None, help='Specify the RNN model suffix name if any.\n Default value is None.')
    parser.add_argument('--train_file_suffix', type=str, default=None, help='Specify the suffix of the file from which train data to be read.\n Default value is None.')
    parser.add_argument('--test_file_suffix', type=str, default=None, help='Specify the suffix of the file from which test data to be read/for saving the test data if needs to be generated.\n Default Value is None.')
    parser.add_argument('--num_examples', type=int, default=100,
                        help='Specify the total number of sequences to generate a balanced test dataset if needed.\n Default value is 100.')

    args = parser.parse_args()
    lang = args.lang
    train_file_suffix = args.train_file_suffix
    test_file_suffix = args.test_file_suffix
    num_examples = args.num_examples
    model_suffix_name = args.model_suffix_name

    if model_suffix_name is None:
        model_suffix_name = ""

    model_name = "modelRNNQ_lang" + str(lang) + "_" + model_suffix_name + ".pt"
    print(f"Information:\n Lang selected: {lang}\n number of samples for test data: {num_examples}\n "
             f"file suffix for test data: {train_file_suffix}\n file suffix for test data: {test_file_suffix}\n"
          f"model_name: {model_name}")


    rnn_model = RNNModel(input_size=input_size, hidden_size=hidden_size,
                         output_size=output_size, num_layers=num_layers, model_name=model_name)

    # Load the model if saved already in the models folder
    RNNModelPath = os.path.join(model_dir, model_name)
    needTraining = rnn_model.load_RNN_model(RNNModelPath)
    debug(0, f"Info: RNNModelPath: {RNNModelPath} and model exists: {os.path.exists(RNNModelPath)}")

    if needTraining:
        X_train, y_train, X_eval, y_eval, num_of_pos_examples_generated, num_of_neg_examples_generated = \
            get_available_datasets(lang, validation=True, train_file_suffix=train_file_suffix)
        print(f"Info: Length of X(input) for training: {len(X_train)}")
        print(f"Info: Size of y(label) tensor for training: {y_train.size()}")
        print(f"Info: pos examples: {num_of_pos_examples_generated}")
        print(f"Info: neg examples: {num_of_neg_examples_generated}")
        rnn_model.train_RNN(X_train, y_train, num_epochs, batch_size, learning_rate, X_eval, y_eval, model_dir=model_dir)
        print(f"Training Done!!")
    else:
        print(f"Info: Training is skipped!")

    if test_file_suffix is None:
        test_file_suffix = ""
    X_test, y_test,  num_of_pos_examples_generated, num_of_neg_examples_generated = \
        generate_test_data(lang, num_examples, maxlength=maxlength, train_file_suffix= train_file_suffix, \
                           test_file_suffix=test_file_suffix, generate_diff_data_from_train=True)
    debug(4, f"X_test:{X_test}")
    numSamples_generated = len(X_test)
    print(f"Info: Length of X(input) for testing: {len(X_test)}")
    print(f"Info: Size of y(label) tensor for testing: {y_test.size()}")
    predicted = rnn_model.test_RNN(X_test, y_test)
    y_test = rnn_model.convertTensor1DTo2D(y_test).numpy()
    debug(4, f"{y_test} and {predicted}")
    pos, neg = rnn_model.checkAccuracy(predicted=predicted, actual=y_test)
    statistics(num_of_pos_examples_generated, num_of_neg_examples_generated, pos, neg)
    rnn_model.getScores(predicted, y_test)