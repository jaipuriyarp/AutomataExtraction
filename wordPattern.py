import os
import psutil
import time
import warnings
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gensim.models import Word2Vec, KeyedVectors

verbose = 0
maxlength = 10
def load_model():
    word2vecPath = "./models/GoogleNews-vectors-negative300.bin"
    warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
    if not os.path.exists(word2vecPath):
        raise Exception("word2vec model doesn't exist at the path:" + str(word2vecPath))
    print(f"Model at {word2vecPath}")

    if (verbose==2):
        process = psutil.Process(os.getpid())
        pre = process.memory_info().rss
        print("Memory used in GB before Loading the Model: %0.2f" % float(pre / (10 ** 9)))  # Check memory usage before loading the model
        print('-' * 10)

        start_time = time.time()  # Start the timer
        # mem= psutil.virtual_memory()
        # ttl = mem.total  # Toal memory available

    w2v_model = KeyedVectors.load_word2vec_format(word2vecPath, binary=True, limit=20000)  # load the model
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

        print("Number of words in vocabs: ", len(w2v_model))  # Number of words in the vocabulary
    return w2v_model
def test_model(w2v_model):
    # word = w2v_model[100]
    word = random.choices(w2v_model.index_to_key)
    # array_for_hash = w2v_model.get_vector("#")
    # print(array_for_hash)
    if not(word):
        raise Exception("w2v Model not loaded properly!")
    print ("w2v Model loading tested Successfully!")

def genSpecialNegExample(num_examples):
    # TODO: create examples without '#' as well
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
            wordL.append([word[0] if i % 2 ==0  else '#' for i in range(k)])
        else:
            x = []
            for i in range(k):
                word = random.choices(w2v_model.index_to_key)
                x.append(word[0])
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
    print("positive examples:", posL)
    print("negative examples:", negL)
    return posL, negL

def create_datasets(num_examples, train=False):
    X = []
    y = []
    posL, negL = generateExamples(num_examples)
    result = np.random.choice(np.concatenate([posL, negL]), num_examples, replace=False)
    for word in result:
        X.append(encode_word(word))
        y.append(1 if word in posL else 0)
    print("Total number of samples generated:", num_examples)
    if train:
        print("Number or positive examples in training:", y.count(1))
        print("Number or negative examples in training:", y.count(0))
    else:
        print("Number or positive examples in test:", y.count(1))
        print("Number or negative examples in test:", y.count(0))
    return X, torch.tensor(y, dtype=torch.float)

if __name__ == "__main__":
    w2v_model = load_model()
    test_model(w2v_model)
    X_train, y_train = create_datasets(10, train=True)
    #X_test, y_test = generate_examples(1000)
