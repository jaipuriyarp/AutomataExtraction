'''
Usage: This file is used for generating the training data for languages 1 till 7.
Info: L1 to L7 is the Tomita extended languages (defined already in thesis).
Python version: Python 3.9.0
How to run:
    python <file_name> --lang <language number in integer> --num_examples <number of examples in integer>
    E.g. py dataGeneratorOnQ.py --lang 7 --num_examples 200
    OR
    python <file_name> --lang <language number in integer> --num_examples <number of examples in integer> --file_suffix <string>
    E.g. py dataGeneratorOnQ.py --lang 7 --num_examples 200 --file_suffix train_data
Default value:
    file_suffix: None : the value used to identify the name of files in which data is saved.
    verbose: 0
Note:
    You can change the following values, which are not user arguments, as per requirements:
    'maxlength' (present in main function):  the maximum length of the sequences to generate
    'start' and 'maxInteger': the start and end range of elements of sequences respectively.
    'data_dir': the directory name to which the generated dataset is saved to/read from.
    'verbose' : value for debugging purposes, by default it is set to 0.
'''

import os
import random
import argparse
from groundTruthFunctions import *

start = 0
maxInteger = (2^63 -1)
data_dir = "../dataOnQ"
verbose = 0

def debug(verbose_level, str):
    if verbose >= verbose_level:
        print(str)

def getGndLabelFunc(lang:int):
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
        raise Exception("No such languages!")

    return checkGndLabel

def call_random(start=0, end=maxInteger):
    return random.randint(start, end)

def genSpecialPosExample(num_examples: int, lang: int, maxlength: int):
    if not(lang == 2 or lang == 4):
        return []
    wordL = set()

    checkGndLabel = getGndLabelFunc(lang)
    if lang == 2:
        fixedlength, wordCounter = 2, 0
        while len(wordL) < num_examples:
            word = generate_two_random_words()
            if wordCounter == 40:
                wordCounter = 0
                fixedlength += 2
                if fixedlength > maxlength:
                    fixedlength = 2
            if checkGndLabel([word[0] if i % 2 == 0 else word[1] for i in range(fixedlength)], False):
                wordL.add(tuple([word[0] if i % 2 == 0 else word[1] for i in range(fixedlength)]))
                wordCounter += 1

    else:
        while len(wordL) < num_examples:
            k = call_random(2, maxlength)
            word = call_random(start, maxInteger)
            validCheck = True
            if lang == 4:
                x = [call_random(start, maxInteger) for _ in range(k - 2)]
                if len(x) > 1:
                    index_to_insert_at = call_random(0, len(x) - 1)
                else:
                    index_to_insert_at = 0
                for i in range(2):
                    x.insert(index_to_insert_at + i, word)
                for i in range(len(x)):
                    if i > 2 and x[i-1] == x[i-2] and x[i] == x[i-1]:
                        validCheck = False
                        break
            if validCheck and checkGndLabel(x, False):
                wordL.add(tuple(x))
    debug(1, f"Special Positive Examples : {wordL}")
    return wordL


def genSpecialNegExample(num_examples: int, lang : int, maxlength: int):
    if lang > 3  and lang < 7:
        return []

    wordL = set()
    checkGndLabel = getGndLabelFunc(lang)
    if lang == 2:
        #length 1 is negative example
        fixedlen = 1
        for i in range(500):
            randNum = call_random(start, maxInteger)
            for j in range(10):
                if not(checkGndLabel([randNum for _ in range(j)], False)):
                    wordL.add(tuple([randNum for _ in range(j)]))

    num_examples -= len(wordL)

    while len(wordL) < num_examples:
        k = call_random(1, maxlength-1)
        word = generate_two_random_words()

        if lang == 1:
            x = [word[0] for _ in range(k)]
            x.insert(call_random(0,len(x)), word[1])

        elif lang == 2:
            if k % 2 != 0:
                k += 1
            if k >= maxlength:
                k -= 2
            x = [word[0] if i%2 == 0 else word[1] for i in range(k)]
            if len(x) -1 == 0:
                index = 0
            else:
                index = call_random(0, len(x) - 1)
            x.insert(index, call_random(start, maxInteger))

        elif lang == 3:
            # odd # of a followed by 0 or odd number of b's.
            if k%2 == 0:
                k += 1 #odd
            x = [word[0] for _ in range(k)]
            k = call_random(0, maxlength-len(x))
            if k % 2 == 0:
                k -= 1
            x += [word[1] for _ in range(k)]

        elif lang == 7:
            # negative case of :a*b*a*b*
            newWord = call_random(start, maxInteger)
            k = call_random(1, maxlength-5)
            x = [word[0] for _ in range(k)]
            if maxlength - 4 - len(x) > 1:
                k = call_random(1, maxlength - 4 - len(x))
                x  += [word[1] for _ in range(k)]
            if maxlength - 3 - len(x) > 1:
                k = call_random(1, maxlength - 3 - len(x))
                x += [word[1] for _ in range(k)]
            if maxlength - 2 - len(x) > 1:
                k = call_random(1, maxlength - 2 - len(x))
                x += [word[1] for _ in range(k)]
            x.insert(call_random(0, len(x) - 1), newWord) # a*b*a*b*(c/a)
            validCheck = (len(set(x)) > 2)
            if validCheck:
                wordL.add(tuple(x))

        if not(checkGndLabel(x, False)):
            wordL.add(tuple(x))
    debug(1, f"special Negatives:{wordL}")
    return wordL

def find_numbers_with_constraint(maxlength: int):
    while True:
        a = random.randint(1, maxlength)
        b = random.randint(1, a)
        x = (a-b) / 3
        if x.is_integer():
            return a, b

def generate_two_random_words():
    word = [call_random(start, maxInteger) for _ in range(2)]
    while word[0] == word[1]:
        word = [call_random(start, maxInteger) for _ in range(2)]
    return word
def generateTypeExamples(num_examples: int, lang: int, pos : bool, maxlength: int):
    wordL = set()
    # lang = 1: L1 = (a)^n.
    # lang = 2: L2 = (ab)^n
    # lang = 3: L3 = (a)^n(b)^m where n is odd and m is even.
    # lang = 4: L4 = any string without Trigram.
    # lang = 5: L5 = any string containing ab and number of occurrence of a and b are even.
    # lang = 6: L6 = (number of a) equivalent to (number of b) mod 3. i.e. 3|(#a - #b)
    # lang = 7: L7 = a^*b^*a^*b^*
    if lang > 7 or lang < 0:
        raise Exception("No such language exists!")

    checkGndLabel = getGndLabelFunc(lang)

    while len(wordL) < num_examples:
        debug(0, f"To more gen: {num_examples-len(wordL)}, len:{len(wordL)}")
        k = call_random(1, maxlength)
        if pos:
            word = generate_two_random_words()
            if lang == 1: # L = (a)^n
                random_numbers = set(np.random.randint(low=0, high=maxInteger, size=num_examples))
                if num_examples < 10:
                    random_numbers = set(np.random.randint(low=0, high=maxInteger, size=num_examples*100))
                idx = 0
                while len(random_numbers) < num_examples:
                    random_numbers.add(int(random.choice(list(random_numbers))*call_random(0, maxInteger)/23))
                    if num_examples < 10:
                        random_numbers.add(int(random.choice(list(random_numbers)) * call_random(2000, maxInteger) / 37))
                    idx+=1
                    if idx == len(random_numbers)-1:
                        idx = 0
                # print(len(random_numbers))
                for x in random_numbers:
                    if len(wordL) > num_examples:
                        break
                    k = call_random(0, maxlength)
                    if checkGndLabel([x for _ in range(k)], False):
                        wordL.add(tuple([x for _ in range(k)]))
            elif lang == 2: # L = (ab)^n
                random_numbers = set(np.random.randint(low=0, high=maxInteger, size=num_examples))
                while len(random_numbers) < int(num_examples/2):
                    random_numbers.add(int(random.choice(list(random_numbers))*call_random(0, maxInteger)/23))
                # print(len(random_numbers))
                random_numbers = list(random_numbers)
                for x in random_numbers:
                    if len(wordL) > num_examples:
                        break
                    k = call_random(0, maxlength)
                    if k % 2 != 0:
                        k -= 1  # even
                    fixed_idx = call_random(0, len(random_numbers)-1)
                    while random_numbers[fixed_idx] == x:
                        fixed_idx = call_random(0, len(random_numbers))
                    if checkGndLabel([x if i % 2 == 0 else random_numbers[fixed_idx] for i in range(k)], False):
                        wordL.add(tuple([x if i % 2 == 0 else random_numbers[fixed_idx] for i in range(k)]))
            elif lang == 3: # L = odd numbers of a must be followed by even numbers of b
                random_numbers = set(np.random.randint(low=0, high=maxInteger, size=num_examples))
                while len(random_numbers) < int(num_examples/2):
                    random_numbers.add(int(random.choice(list(random_numbers))*call_random(0, maxInteger)/23))
                # print(len(random_numbers))
                for _ in random_numbers:
                    if len(wordL) > num_examples:
                        break
                    word = random.sample(list(random_numbers), 2)
                    k = call_random(1, maxlength-2)
                    if k%2 == 0:
                        k -= 1 # odd k
                    x = [word[0] for _ in range(k)]
                    k2 = call_random(2, maxlength -len(x))
                    if k2 % 2 != 0:
                        k2 -= 1
                    x +=  [word[1] for _ in range(k2)]
                    if checkGndLabel(x, False):
                        wordL.add(tuple(x))

            elif lang == 4: # L = any string not containing aaa (3a's consecutively)
                x = []
                for i in range(k):
                    p = call_random(start, maxInteger)
                    if i > 1 and x[i-1] == x[i-2]:
                        while x[i-2] == p:
                            p = call_random(start, maxInteger)
                    x.append(p)
                if checkGndLabel(x, False):
                    wordL.add(tuple(x))
            elif lang == 5: # L = number of a and number of b in string are even.
                if k % 2 == 1:
                    k -= 1 # even
                x = [word[call_random(0, 1)] for _ in range(k-2)]
                if x.count(word[0]) % 2 == 1:
                    x.append(word[0])
                if x.count(word[1]) % 2 == 1:
                    x.append(word[1])
                if checkGndLabel(x, False):
                    wordL.add(tuple(x))
            elif lang == 6: # L = #a equivalent #b mod 3. i.e. 3|(#a - #b) where
             # #x=: number of x in string a|b := a divides b.
                random_numbers = set(np.random.randint(low=0, high=maxInteger, size=num_examples))
                while len(random_numbers) < int(num_examples):
                    random_numbers.add(int(random.choice(list(random_numbers)) * call_random(0, maxInteger) / 23))
                # print(len(random_numbers))
                random_numbers = list(random_numbers)
                for _ in random_numbers:
                    if len(wordL) > num_examples:
                        break
                    word = random.sample(random_numbers, 2)
                    num_a, num_b = find_numbers_with_constraint(maxlength)
                    x = [word[0] for _ in range(num_a)]
                    for _ in range(num_b):
                        index_to_insert_at = call_random(0, len(x) - 1)
                        x.insert(index_to_insert_at, word[1])
                    validCheck = x.count(word[0]) - x.count(word[1]) % 3 == 0
                    if validCheck and checkGndLabel(x, False):
                        wordL.add(tuple(x))
            elif lang == 7: #a^*b^*a^*b^*
                x = [word[0] for _ in range(k)]
                if maxlength - len(x) > 1:
                    x += [word[1] for _ in range(call_random(1, maxlength - len(x)))]
                if maxlength - len(x) > 1:
                    x += [word[0] for _ in range(call_random(1, maxlength - len(x)))]
                if maxlength - len(x) > 1:
                    x += [word[0] for _ in range(call_random(1, maxlength - len(x)))]
                if checkGndLabel(x, False):
                    wordL.add(tuple(x))

        else: # neg case
            if lang == 1:
                x = [call_random(0, maxInteger) for _ in range(k)]
                validCheck = (k > 1 and len(set(x)) > 1)
            elif lang == 2:
                x = [call_random(start, maxInteger) for _ in range(k)]
                validCheck = (k%2 == 1) or (k > 2 and len(set(x)) > 2)
            elif lang == 3:
                # neg case 1: even # of a followed by even # of b (includes number of b to be zero)
                # neg case 2: even # of a followed by odd  # of b
                word = generate_two_random_words()
                if k % 2 == 1:
                    k -= 1
                x = [word[0] for _ in range(k)]
                x += [word[1] for _ in range(call_random(0, maxlength-len(x)))]
                validCheck = x.count(word[0]) %2 == 0
            elif lang == 4:
                # neg case: a word whose length is > 2 and at least contains trigram once.
                x = [call_random(start, maxInteger) for _ in range(k-3)]
                if len(x):
                    index_to_insert_at = call_random(0, len(x)-1)
                else:
                    index_to_insert_at = 0
                repeated_word = call_random(start, maxInteger)
                for i in range(3):
                    x.insert(index_to_insert_at + i, repeated_word)
                validCheck = x[index_to_insert_at] == x[index_to_insert_at + 1] and \
                             x[index_to_insert_at] == x[index_to_insert_at + 1]
            elif lang == 5:
                # neg case: either number of a/b is odd or both a and b are odd.
                word = generate_two_random_words()
                x = [word[call_random(0, 1)] for _ in range(k)]
                if x.count(word[0]) % 2 == 0 and x.count(word[1]) % 2 == 0:
                    del x[0]
                validCheck = not (x.count(word[0]) % 2 == 0 and x.count(word[1]) % 2 == 0)
            elif lang == 6:
                word = generate_two_random_words()
                x = [word[0] for _ in range(k)]
                k2 = call_random(0, maxlength - len(x))
                x += [word[1] for _ in range(k2)]
                while (x.count(word[0]) - x.count(word[1])) % 3 == 0:
                    del x[call_random(0, len(x)-1)]
                validCheck = not(x.count(word[0]) - x.count(word[1]) % 3 == 0)
            elif lang == 7:
                x = [call_random(start, maxInteger) for _ in range(k)]
                validCheck = (k > 2 and len(set(x)) > 2)

            if validCheck and (not checkGndLabel(x, False)):
                wordL.add(tuple(x))

    return wordL


def generateExampleOfEachlength(lang: int, maxlength: int):
    pos, neg = set(), set()
    checkGndLabel = getGndLabelFunc(lang)
    for l in range(0, maxlength):
        w = generate_two_random_words()
        for sublist_length in range(l):
            sublist = tuple([w[random.randint(0, 1)] for _ in range(sublist_length)])
            if checkGndLabel(sublist, False):
                pos.add(sublist)
            else:
                neg.add(sublist)
    return pos, neg


def generateExamples(num_examples: int, lang: int, maxlength: int) -> list:
    posL, negL = generateExampleOfEachlength(lang, maxlength)

    posL_count = int(num_examples / 2) - len(posL)
    negL_count = num_examples - int(num_examples / 2) - len(negL)

    negL = negL.union(genSpecialNegExample(int(negL_count / 2), lang, maxlength))
    posL = posL.union(genSpecialPosExample(int(posL_count / 2), lang, maxlength))
    debug(0, f"Number of special +ve data points: {len(posL)}")
    debug(0, f"Number of special -ve data points: {len(negL)}")
    negL_count = negL_count - len(negL)
    posL_count = posL_count - len(posL)

    if posL_count > 0:
        while len(set(posL)) < int(num_examples/2):
            posL = posL.union(generateTypeExamples(posL_count, lang, pos=True, maxlength=maxlength))
    if negL_count > 0:
        while len(set(negL)) < int(num_examples/2):
            negL = negL.union(generateTypeExamples(negL_count, lang, pos=False, maxlength=maxlength))

    debug(0, f"Number of +ve data points: {len(posL)}")
    debug(0, f"Number of -ve data points: {len(negL)}")
    return list(posL), list(negL)


def create_datasets(num_examples: int, lang: int, maxlength: int, file_suffix: str):
    posL, negL = generateExamples(num_examples, lang=lang, maxlength=maxlength)
    debug(0, f"length of pos examples:{len(posL)} and length of neg examples: {len(negL)}")

    file_name = "pos_lang" + str(lang)
    if file_suffix != "":
        file_name = file_name + "_" + file_suffix

    file_path = os.path.join(data_dir, file_name)
    with open(file_path, "w+") as f:
        f.write("len:" + str(len(posL)) + "\n")
        f.write(str(posL))
    debug(0, f"File saved:{file_path}")

    # file_name_for_savingdata = "neg_" + file_suffix + "_lang" + str(lang)
    # file_name_for_savingdata = "neg_lang" + str(lang) + "_" + file_suffix
    file_name = "neg_lang" + str(lang)
    if file_suffix != "":
        file_name = file_name + "_" + file_suffix
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, "w+") as f:
        f.write("len:" + str(len(negL)) + "\n")
        f.write(str(negL))
    debug(0, f"File saved:{file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This file is used for creating a balanced dataset for languages 1 to 7 (as per extended Tomita languages definition).')
    parser.add_argument('--lang', type=int, help='Specify the language from 1 to 7, for which to create a balanced data.')
    parser.add_argument('--file_suffix', type=str, default=None, help='Specify the suffix of the file for saving the data.\n Default value is None.')
    parser.add_argument('--num_examples', type=int, help='Specify the total number of sequences to generate a balanced dataset.')

    args = parser.parse_args()
    lang = args.lang
    file_suffix = args.file_suffix
    num_examples = args.num_examples

    maxlength = 20
    debug(0, f"Information:\n Lang selected: {lang}\n number of samples: {num_examples}\n file_suffix: {file_suffix}\n maximum length: {maxlength}")
    if file_suffix is None:
        file_suffix = ""
    create_datasets(num_examples, lang=lang, maxlength=maxlength, file_suffix=file_suffix)
    debug(0, f"Data Generation is complete!")