import random
import torch
import numpy as np
from rnnModel import  RNNModel

# changing the languages according to Tomita languages

verbose = 0
maxlength = 50
maxInteger = (2^63 -1)

# Define hyperparameters
input_size = 1
hidden_size = 64
output_size = 1
num_layers = 2

# Define hyperparameters
learning_rate = 0.001
num_epochs = 500
batch_size = 32
model_name = "modelRNNQ_lang7_aStarbStaraStarbStarSeq.pt"
lang = 7

def debug(verbose_level, str):
    if verbose >= verbose_level:
        print(str)


def genSpecialPosExample(num_examples, lang):
    if lang != 4:
        return []
    wordL = []
    while num_examples > 0:
        k = random.randint(2, maxlength)
        word = random.randint(0, maxInteger)
        validCheck = True
        if lang == 4:
            x = [random.randint(0, maxInteger) for _ in range(k - 2)]
            if len(x) > 1:
                index_to_insert_at = random.randint(0, len(x) - 1)
            else:
                index_to_insert_at = 0
            for i in range(2):
                x.insert(index_to_insert_at + i, word)
            for i in range(len(x)):
                if i > 2 and x[i-1] == x[i-2] and x[i] == x[i-1]:
                    validCheck = False
                    break
        if validCheck:
            wordL.append(x)
        else:
            num_examples += 1 # since the current word is not added
        num_examples -= 1
    debug(1, f"Special Positive Examples : {wordL}")
    return wordL


def genSpecialNegExample(num_examples, lang):
    if lang > 3  and lang < 7:
        return []

    wordL = []
    while num_examples > 0:
        k = random.randint(1, maxlength-1)
        word = generate_two_random_words()

        if lang == 1:
            x = [word[0] for _ in range(k)]
            x.insert(random.randint(0,len(x)), word[1])

        elif lang == 2:
            if k % 2 != 0:
                k += 1
            if k >= maxlength:
                k -= 2
            if word[0] > word[1]:
                word.reverse()
            x = [word[0] if i%2 == 0 else word[1] for i in range(k)]
            if k > 1:
                newWord = random.randint(0, maxInteger)
                word.append(newWord)
                x.insert(random.randint(0, len(x)-1), word[random.randint(0,len(word)-1)])
            #creating one more neg example here: case: (ab)^n but a>b
            y = [word[1] if i%2 == 0 else word[0] for i in range(k)]
            wordL.append(y)
            num_examples -= 1

        elif lang == 3:
            # odd # of a followed by 0 or odd number of b's.
            if k%2 == 0:
                k += 1 #odd
            x = [word[0] for _ in range(k)]
            k = random.randint(0, maxlength-len(x))
            if k % 2 == 0:
                k -= 1
            x += [word[1] for _ in range(k)]

        elif lang == 7:
            # negative case of :a*b*a*b*
            newWord = random.randint(0, maxInteger)
            k = random.randint(1, maxlength-5)
            x = [word[0] for _ in range(k)]
            if maxlength - 4 - len(x) > 1:
                k = random.randint(1, maxlength - 4 - len(x))
                x  += [word[1] for _ in range(k)]
            if maxlength - 3 - len(x) > 1:
                k = random.randint(1, maxlength - 3 - len(x))
                x += [word[1] for _ in range(k)]
            if maxlength - 2 - len(x) > 1:
                k = random.randint(1, maxlength - 2 - len(x))
                x += [word[1] for _ in range(k)]
            x.insert(random.randint(0, len(x) - 1), newWord) # a*b*a*b*(c/a)
            validCheck = (len(set(x)) > 2)
            if validCheck:
                wordL.append(x)

        wordL.append(x)
        num_examples -= 1
    debug(1, f"special Negatives:{wordL}")
    return wordL

def generate_two_random_words():
    word = [random.randint(0, maxInteger) for _ in range(2)]
    while word[0] == word[1]:
        word = [random.randint(0, maxInteger) for _ in range(2)]
    return word
def generateTypeExamples(num_examples, lang, pos):
    wordL = []
    # lang = 1: L1 = (a)^n.
    # lang = 2: L2 = (ab)^n and a<b.
    # lang = 3: L3 = (a)^n(b)^m where a is odd and b is even.
    # lang = 4: L4 = any string without Trigram.
    # lang = 5: L5 = any string containing ab and number of occurence of a and b are even.
    # lang = 6: L6 = (number of a) equivalent to (number of b) mod 3. i.e. 3|(#a - #b)
    # lang = 7: L7 = a^*b^*a^*b^*
    if lang > 7 or lang < 0:
        raise Exception("No such language exists!")

    while num_examples > 0:
        k = random.randint(1, maxlength)
        if pos:
            word = generate_two_random_words()
            if lang == 1: # L = (a)^n
                k = random.randint(0, maxlength)
                wordL.append([word[0] for _ in range(k)])
            elif lang == 2: # L = (ab)^n and a<b
                if k % 2 != 0:
                    k -= 1 # even
                if word[0] > word[1]:
                    word.reverse()
                wordL.append([word[0] if i % 2 == 0 else word[1] for i in range(k)])
            elif lang == 3: # L = odd numbers of a must be followed by even numbers of b
                k = random.randint(1, maxlength-2)
                if k%2 == 0:
                    k -= 1 # odd k
                x = [word[0] for _ in range(k)]
                k2 = random.randint(2, maxlength -len(x))
                if k2 % 2 != 0:
                    k2 -= 1
                x +=  [word[1] for _ in range(k2)]
                wordL.append(x)
            elif lang == 4: # L = any string not containing aaa (3a's consecutively)
                x = []
                for i in range(k):
                    p = random.randint(0, maxInteger)
                    if i > 1 and x[i-1] == x[i-2]:
                        while x[i-2] == p:
                            p = random.randint(0, maxInteger)
                    x.append(p)
                wordL.append(x)
            elif lang == 5: # L = number of a and number of b in string are even.
                if k % 2 == 1:
                    k -= 1 # even
                x = [word[random.randint(0, 1)] for _ in range(k-2)]
                if x.count(word[0]) % 2 == 1:
                    x.append(word[0])
                if x.count(word[1]) % 2 == 1:
                    x.append(word[1])
                wordL.append(x)
            elif lang == 6: # L = #a equivalent #b mod 3. i.e. 3|(#a - #b) where
             # #x=: number of x in string a|b := a divides b.
                if maxlength - k > 0:
                    k2 = random.randint(0, maxlength-k)
                else:
                    k2 = 0
                while k - k2 % 3 != 0:
                    if k - 1 >= 0:
                        k -= 1
                    else:
                        k2 -= 1
                x = [word[0] for _ in range(k)]
                if len(x) - 1 >= 0:
                    index_to_insert_at = random.randint(0, len(x)-1)
                else:
                    index_to_insert_at = 0
                for _ in range(k2):
                    x.insert(index_to_insert_at, word[1])
                validCheck = x.count(word[0]) - x.count(word[1]) % 3 == 0
                if validCheck:
                    wordL.append(x)
                else:
                    num_examples += 1
            elif lang == 7: #a^*b^*a^*b^*
                x = [word[0] for _ in range(k)]
                if maxlength - len(x) > 1:
                    x += [word[1] for _ in range(random.randint(1, maxlength - len(x)))]
                if maxlength - len(x) > 1:
                    x += [word[0] for _ in range(random.randint(1, maxlength - len(x)))]
                if maxlength - len(x) > 1:
                    x += [word[0] for _ in range(random.randint(1, maxlength - len(x)))]
                wordL.append(x)

        else: # neg case
            if lang == 1:
                x = [random.randint(0, maxInteger) for _ in range(k)]
                validCheck = (k > 1 and len(set(x)) > 1)
            elif lang == 2:
                x = [random.randint(0, maxInteger) for _ in range(k)]
                validCheck = (k%2 == 1) or (k > 2 and len(set(x)) > 2) or (k==2 and x[0]>x[1])
            elif lang == 3:
                # neg case 1: even # of a followed by even # of b (includes number of b to be zero)
                # neg case 2: even # of a followed by odd  # of b
                word = generate_two_random_words()
                if k % 2 == 1:
                    k -= 1
                x = [word[0] for _ in range(k)]
                x += [word[1] for _ in range(random.randint(0, maxlength-len(x)))]
                validCheck = x.count(word[0]) %2 == 0
            elif lang == 4:
                # neg case: a word whose length is > 2 and at least contains trigram once.
                x = [random.randint(0, maxInteger) for _ in range(k-3)]
                if len(x):
                    index_to_insert_at = random.randint(0, len(x)-1)
                else:
                    index_to_insert_at = 0
                repeated_word = random.randint(0, maxInteger)
                for i in range(3):
                    x.insert(index_to_insert_at + i, repeated_word)
                validCheck = x[index_to_insert_at] == x[index_to_insert_at + 1] and \
                             x[index_to_insert_at] == x[index_to_insert_at + 1]
            elif lang == 5:
                # neg case: either number of a/b is odd or both a and b are odd.
                word = generate_two_random_words()
                x = [word[random.randint(0, 1)] for _ in range(k)]
                if x.count(word[0]) % 2 == 0 and x.count(word[1]) % 2 == 0:
                    del x[0]
                validCheck = not (x.count(word[0]) % 2 == 0 and x.count(word[1]) % 2 == 0)
            elif lang == 6:
                word = generate_two_random_words()
                x = [word[0] for _ in range(k)]
                k2 = random.randint(0, maxlength - len(x))
                x += [word[1] for _ in range(k2)]
                while (x.count(word[0]) - x.count(word[1])) % 3 == 0:
                    del x[random.randint(0, len(x)-1)]
                validCheck = not(x.count(word[0]) - x.count(word[1]) % 3 == 0)
            elif lang == 7:
                x = [random.randint(0, maxInteger) for _ in range(k)]
                validCheck = (k > 2 and len(set(x)) > 2)

            if validCheck:
                wordL.append(x)
            else:
                num_examples += 1 # since x is not added in the wordL
        num_examples -= 1
    return wordL

def generateExamples(num_examples: int, lang: int) -> list:

    posL_count = int(num_examples / 2)
    negL_count = num_examples - posL_count
    negL = genSpecialNegExample(int(negL_count / 2), lang)
    posL = genSpecialPosExample(int(posL_count / 2), lang)
    debug(1, f"Number of special +ve data points: {len(posL)}")
    debug(1, f"Number of special -ve data points: {len(negL)}")
    negL_count = negL_count - len(negL)
    posL_count = posL_count - len(posL)
    if posL_count != 0:
        posL = posL + generateTypeExamples(posL_count, lang, pos=True)
    if negL_count != 0:
        negL = negL + generateTypeExamples(negL_count, lang, pos=False)
    debug(1, "positive examples:" + str(posL))
    debug(1, "negative examples:" + str(negL))
    return posL, negL

def encode_sequence(sequence):
    debug(2, f"sequence: {sequence}")
    x = torch.tensor(np.array([[word] for word in sequence]))
    debug(2, "size of x: {x.size()}")
    target_seq = torch.zeros(maxlength, input_size)
    if x.size()[0] > 0:
        target_seq[0:x.size(0),:] = x
    debug(2, "size of target: {target_seq.size()}")
    debug(2, "target: {target_seq}")
    return target_seq

def create_datasets(num_examples, lang, train=False):
    X = []
    y = []

    posL, negL = generateExamples(num_examples, lang=lang)
    result = posL + negL
    random.shuffle(result)
    debug(3, f"result: {result}")
    for sequence in result:
        X.append(encode_sequence(sequence))
        y.append(1 if sequence in posL else 0)
    debug(3, "X =" + str(X))
    debug(3, "y=" + str(y))
    print(f"Info: Total number of samples generated: {num_examples}")
    if train:
        print(f"Info: Number of positive examples in training: {y.count(1)}")
        print(f"Info: Number of negative examples in training: {y.count(0)}")
    else:
        print(f"Info: Number of positive examples in test: {y.count(1)}")
        print(f"Info: Number of negative examples in test: {y.count(0)}")
    return X, torch.tensor(y, dtype=torch.float)

def statistics(numSamples, pos, neg):

    print(f"Statistics: Accuracy %                                           : {(pos + neg) / numSamples}")
    print(
        f"Statistics: Inaccuracy %                                         : {(numSamples - (pos + neg)) / numSamples}")
    print(f"Statistics: Total number of data points                          : {numSamples}")
    print(f"Statistics: Total number of data points classified correctly     : {pos + neg}")
    print(f"Statistics: Total number of data points misclassified            : {numSamples - (pos + neg)}")
    print(f"Statistics: Total number of +ve data points identified correctly : {pos}")
    print(f"Statistics: Total number of -ve data points identified correctly : {neg}")
    print(f"Statistics: Total +ve Accuracy %                                 : {pos / int(numSamples / 2)}")
    print(f"Statistics: Total -ve Accuracy %                                 : {neg / (numSamples - int(numSamples / 2))}")


if __name__ == "__main__":

    rnn_model = RNNModel(input_size=input_size, hidden_size=hidden_size,
                         output_size=output_size, num_layers=num_layers, model_name=model_name)

    # Load the model saved already in the models folder
    RNNModelPath = "../models/" + model_name
    needTraining = rnn_model.load_RNN_model(RNNModelPath)


    if needTraining:
        numSamples = 300000
        X_train, y_train = create_datasets(numSamples, lang=lang, train=True)
        print(f"Info: Length of X(input) for training: {len(X_train)}")
        print(f"Info: Size of y(label) tensor for training: {y_train.size()}")
        rnn_model.train_RNN(X_train, y_train, num_epochs, batch_size, learning_rate)
    else:
        print(f"Info: Training is skipped!")

    numSamples = 90000

    X_test, y_test = create_datasets(numSamples, lang=lang, train=False)
    print(f"Info: Length of X(input) for testing: {len(X_test)}")
    print(f"Info: Size of y(label) tensor for testing: {y_test.size()}")
    predicted = rnn_model.test_RNN(X_test, y_test)
    y_test = rnn_model.convertTensor1DTo2D(y_test).numpy()
    # print(f"{y_test} and {predicted}")
    pos, neg = rnn_model.checkAccuracy(predicted=predicted, actual=y_test)
    statistics(numSamples, pos, neg)
    rnn_model.getScores(predicted, y_test)