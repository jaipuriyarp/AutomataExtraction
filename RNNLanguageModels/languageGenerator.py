import os
import psutil
import time
import warnings
import random
import torch
import numpy as np

from gensim.models import Word2Vec, KeyedVectors
from rnnModel import  RNNModel

#gensim.__version__ = 4.3.1

verbose = 0
maxlength = 40

# Define hyperparameters
input_size = 300
hidden_size = 64
output_size = 1
num_layers = 2

# Define hyperparameters
learning_rate = 0.001
num_epochs = 800
batch_size = 32
modelName = "modelRNN_abSeq.pt"

def debug(verbose_level, str):
    if verbose >= verbose_level:
        print(str)

# W2V Model:
def load_model(word2vecPath, limit=None):

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

    w2v_model = KeyedVectors.load_word2vec_format(word2vecPath, binary=True, limit=limit)  # load the model
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

    debug(0,f"Info: Number of words in vocabs: {len(w2v_model)}")  # Number of words in the vocabulary
    return w2v_model
def test_model(w2v_model):
    # word = w2v_model[100]
    word = random.choices(w2v_model.index_to_key)
    # array_for_hash = w2v_model.get_vector("#")
    # print(array_for_hash)
    if not(word):
        raise Exception("w2v Model not loaded properly!")
    debug (0, "Info: w2v Model loading tested Successfully!")

def update_w2vModelVocab(newSentence, model):
    pass
#     newModelPath = "/models/w2v_model_updatedVersion.bin"
#     new_words = newSentence
#     # # class SubscriptableWord2Vec(Word2Vec):
#     # #     def __getitem__(self, word):
#     # #         return self.wv[word]
#     #
#     # extended_model = Word2Vec(vector_size=w2v_model.vector_size, min_count=1)
#     # extended_model.build_vocab_from_freq({word: 1 for word in new_words})
#     # extended_model.wv.vectors = np.concatenate([w2v_model.vectors, extended_model.wv.vectors], axis=0)
#     # # extended_model.wv.vectors = extended_vectors
#     # # extended_model.save(newModelPath)
#     # # save_word2vec_format(extended_model, newModelPath, binary=True)
#     # extended_model.wv.save_word2vec_format(newModelPath, binary=True)
#     # # extended_model = KeyedVectors.load(newModelPath)
#
#     # Determine the vector size of the model
#     vector_size = model.vector_size
#
#     # Create random vectors for the new words
#     new_vectors = np.random.random((len(new_words), vector_size))
#
#     # Update the model with new words and vectors
#     model.add_vectors(new_words, new_vectors)
#
#     # Save the updated model
#     model.save(newModelPath)
#     # model.save_word2vec_format(newModelPath, binary=True)
#     return model

def generateNewWord():
    # return a list of new words which is not present in the default vocab of w2v model:
    # word = random.randint(1, 999999999999999999)
    # return ["915977186963600798"]
    # return ["Jaipuriyar"]
    return []
def special_word():
    # change the code for appending any special word for testing RNN
    word = generateNewWord()
    k = random.randint(1, maxlength)
    print(f'special word chosen is : {word} and sentence length is: {k}')
    if k%2 != 0:
        k += 1
    special_sentence = [word if i%2 == 0 else "#" for i in range(k)]
    return special_sentence

def genSpecialNegExample(num_examples, w2v_model, lang):
    # TODO: create examples without '#' as well : Already added due to length selection
    wordL = []
    while num_examples > 0:
        k = random.randint(1, maxlength-2)
        if k % 2 == 1:
            k += 1
        word = random.choices(w2v_model.index_to_key, k=2)
        if lang == 1:
            b  = '#'
        else:
            b = word[1]
        x = [word[0] if i % 2 == 0 else b for i in range(k)]

        word = random.choices(w2v_model.index_to_key, k=2)
        while word[0] == x[0]:
            word = random.choices(w2v_model.index_to_key, k=2)
        x.append(word[0])
        if lang == 2:
            b = random.choices([x[1], word[1]])
        x.append(b[0])
        k = random.randint(0, maxlength-len(x))
        y = [x[0] if i%2 ==0 else b[0] for i in range(k)]
        x = x + y
        wordL.append(x)
        num_examples -= 1
    debug(2, f"special Negatives:{wordL}")
    return wordL
def generateTypeExamples(num_examples, w2v_model, lang, pos):
    wordL = []
    # lang = 1: L1 = (a#)^n
    # lang = 2: L2 = (ab)^n
    if lang > 2 and lang < 1:
        raise Exception("No lang defined!")

    while num_examples > 0:
        k = random.randint(1, maxlength)
        if pos:
            word = random.choices(w2v_model.index_to_key, k=2)
            if k%2 != 0: #To add '# at the end of positive examples if k is -ve
                k += 1
            if lang == 1: # L = (a#)^n
                b = '#'
            elif lang == 2: # L = (ab)^n
                b = word[1]
            wordL.append([word[0] if i%2 == 0  else b for i in range(k)])
        else:
            if lang == 1:
                x = [w2v_model.index_to_key[random.randint(0, len(w2v_model.index_to_key))] if i % 2 == 0 else '#'
                     for i in range(k)]
                validCheck = (k%2 == 1) or (k > 2 and len(set(x)) > 2) or (k==2 and x[0] == '#')
            elif lang == 2:
                # print (f"j1: {j1}, j2: {j2}")
                x = [w2v_model.index_to_key[random.randint(0, len(w2v_model.index_to_key)-1)] if i % 2 == 0 else
                     w2v_model.index_to_key[random.randint(0, len(w2v_model.index_to_key)-1)] for i in range(k)]
                validCheck = (k%2 == 1) or (k > 2 and len(set(x)) > 2) or (k==2 and x[0]==x[1])
            if validCheck:
                wordL.append(x)
            else:
                num_examples += 1 # since x is not added in the wordL
        num_examples -= 1
    return wordL

def generateExamples(num_examples, w2v_model, lang):

    posL_count = int(num_examples / 2)
    negL_count = num_examples - posL_count
    negL = genSpecialNegExample(int(negL_count / 2), w2v_model, lang)
    negL_count = negL_count - int(negL_count / 2)
    posL = []
    if posL_count != 0:
        posL = generateTypeExamples(posL_count, w2v_model, lang, pos=True)
    if negL_count != 0:
        negL = negL + generateTypeExamples(negL_count, w2v_model, lang, pos=False)
    debug(2, "positive examples:" + str(posL))
    debug(2, "negative examples:" + str(negL))
    return posL, negL

def encode_sequence(sequence, w2v_model):
    debug(1, "sequence:" + str(sequence))
    x = torch.tensor(np.array([w2v_model[word] for word in sequence]))
    debug(1, "size of x:" + str(x.size()))
    target_seq = torch.zeros(maxlength,300)
    if x.size()[0] > 0:
        target_seq[0:x.size(0),:] = x
    debug(1, "size of target:" + str(target_seq.size()))
    debug(2, "target:" + str(target_seq))
    return target_seq

def create_datasets(num_examples, w2v_model, lang=1, train=False):
    X = []
    y = []
    # update the w2v_model with new words, if we want to test RNN model on words outside the default vocab of w2v model.
    newWordList = generateNewWord()
    if newWordList != []:
        w2v_model = update_w2vModelVocab(newWordList, w2v_model)

    posL, negL = generateExamples(num_examples, w2v_model, lang=lang)
    result = posL + negL
    random.shuffle(result)
    debug(3, f"result: {result}")
    for sequence in result:
        X.append(encode_sequence(sequence, w2v_model))
        y.append(1 if sequence in posL else 0)
    debug(3, "X =" + str(X))
    debug(3, "y=" + str(y))
    print(f"Info: Total number of samples generated: {num_examples}")
    if train:
        print(f"Info: Number of positive examples in training: {y.count(1)}")
        print(f"Info: Number of negative examples in training: {y.count(0)}")
    else:
        if newWordList != []:
            X.append(encode_sequence(special_word(), w2v_model))
            y.append(1)
        print(f"Info: Number of positive examples in test: {y.count(1)}")
        print(f"Info: Number of negative examples in test: {y.count(0)}")
    return X, torch.tensor(y, dtype=torch.float)

def statistics(numSamples, pos, neg):

    print(f"Statistics: Accuracy %                                           : {(pos + neg) / numSamples}")
    print(
        f"Statistics: Inaccuracy %                                         : {(numSamples - (pos + neg)) / numSamples}")
    print(f"Statistics: Total number of data points                          : {numSamples}")
    print(f"Statistics: Total number of data points classisified correctly   : {pos + neg}")
    print(f"Statistics: Total number of data points misclassisified          : {numSamples - (pos + neg)}")
    print(f"Statistics: Total number of +ve data points identified correctly : {pos}")
    print(f"Statistics: Total number of -ve data points identified correctly : {neg}")
    print(f"Statistics: Total +ve Accuracy %                                 : {pos / int(numSamples / 2)}")
    print(f"Statistics: Total -ve Accuracy %                                 : {neg / (numSamples - (numSamples / 2))}")


if __name__ == "__main__":
    word2vecPath = "../models/GoogleNews-vectors-negative300.bin"
    w2v_model = load_model(word2vecPath)
    test_model(w2v_model)

    RNN_model = RNNModel(input_size=input_size, hidden_size=hidden_size,
                         output_size=output_size, num_layers=num_layers, model_name=modelName)

    # Load the model saved already in the models folder
    RNNModelPath = "../models/" + modelName
    needTraining = RNN_model.load_RNN_model(RNNModelPath)

    lang = 2
    if needTraining:
        X_train, y_train = create_datasets(300000, w2v_model=w2v_model, lang=lang, train=True)
        print(f"Info: Length of X(input) for training: {len(X_train)}")
        print(f"Info: Size of y(label) tensor for training: {y_train.size()}")
        RNN_model.train_RNN(X_train, y_train, num_epochs, batch_size, learning_rate)
    else:
        print(f"Info: Training is skipped!")

    numSamples = 90000

    X_test, y_test = create_datasets(numSamples, w2v_model=w2v_model, lang=lang, train=False)
    print(f"Info: Length of X(input) for testing: {len(X_test)}")
    print(f"Info: Size of y(label) tensor for testing: {y_test.size()}")
    predicted = RNN_model.test_RNN(X_test, y_test)
    y_test = RNN_model.convertTensor1DTo2D(y_test).numpy()
    # print(f"{y_test} and {predicted}")
    pos, neg = RNN_model.checkAccuracy(predicted=predicted, actual=y_test)
    statistics(numSamples, pos, neg)
    RNN_model.getScores(predicted, y_test)