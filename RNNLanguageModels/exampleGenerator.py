import sys
import random
from groundTruthFunctions import *
import itertools
import pandas as pd
# changing the languages according to Tomita languages

pathsToInclude = ["."]
for path in pathsToInclude:
    sys.path.append(path)

from languageGeneratorOnQ import generatelimitedExamples

lang = 1

verbose = 0
maxlength = 20
maxInteger = (2^63 -1)
upper_limit = 15
upper_limit_ex = int(500000/2)
upper_limit_ex_pos = int(500000/4)
upper_limit_ex_neg = int(500000/4)

def debug(verbose_level, str):
    if verbose >= verbose_level:
        print(str)

def generate_n_random_words(n:int):
    end, start = maxInteger, 1
    if n > (end - start + 1):
        raise ValueError("Cannot generate more unique numbers than the range allows")

    unique_numbers = random.sample(range(start, end + 1), n)
    if len(set(unique_numbers)) == n:
        return unique_numbers
    else:
        while(len(set(unique_numbers)) !=n):
            unique_numbers = random.sample(range(start, end + 1), n)
        return unique_numbers


def generate_arrangements(numberlist:list, length:int):
    yield itertools.product(numberlist, repeat=length)
def generateExampleFromLengthOne(upper_limit_ex, lang=0, flag_to_gen=0) -> list:
    #flag_to_gen : if only pos : 1, only neg :2, if both:0
    # lang = 1: L1 = (a)^n.
    # lang = 2: L2 = (ab)^n
    # lang = 3: L3 = (a)^n(b)^m where n is odd and m is even.
    # lang = 4: L4 = any string without Trigram.
    # lang = 5: L5 = any string containing ab and number of occurrence of a and b are even.
    # lang = 6: L6 = (number of a) equivalent to (number of b) mod 3. i.e. 3|(#a - #b)
    # lang = 7: L7 = a^*b^*a^*b^*
    upper_limit_ex_pos = int(upper_limit_ex/2)
    upper_limit_ex_neg = int(upper_limit_ex/2)

    emptyWord = ()
    posL1, negL1 = [], []
    posL2, negL2 = [], []
    posL3, negL3 = [], []
    # posL4, negL4 = [], []
    posL5, negL5 = [], []
    posL6, negL6 = [], []
    posL7, negL7 = [], []

    break_now = False
    for k in range(1, 20):
        # if k%2 == 0:
        print(f"Info: the number of unique number in list is k:{k} ")
        for i in range(1, maxlength+1):
            l = i
            random_words = generate_n_random_words(k)

            if k > 2 and k <= 19 and i > 7:
                if i > 7 and i < 11:
                    l = i - 7
                elif i < 17:
                    l = i - 10
                else:
                    l = i - 13

            if i%5 == 0:
                print(f"Info: The length is i:{i} and l:{l}")

            for y in generate_arrangements(random_words, l):
                for x in y:
                    #lang:2
                    # x = list(x)
                    if len(negL2) >= upper_limit_ex_neg and len(posL2) >= upper_limit_ex_pos:
                        print(f"Going for k:{k} and i:{i} and l :{l} but BREAK")
                        break_now = True
                        break

                    if lang == 0 or lang == 1:
                        if Lang_is_aStar(x, check=False):
                            if (flag_to_gen == 0 or flag_to_gen == 1) and len(posL1) < upper_limit_ex_pos:
                                posL1.append(x)
                        else:
                            if (flag_to_gen == 0 or flag_to_gen == 2) and len(negL1) < upper_limit_ex_neg:
                                negL1.append(x)

                    if lang == 0 or lang == 2:
                        if Lang_is_abSeq(x, check=False):
                            if (flag_to_gen == 0 or flag_to_gen == 1) and len(posL2) < upper_limit_ex_pos:
                                posL2.append(x)
                        else:
                            if (flag_to_gen == 0 or flag_to_gen == 2) and len(negL2) < upper_limit_ex_neg:
                                negL2.append(x)
                    #lang:3
                    if lang == 0 or lang == 3:
                        if Lang_is_abSeq_OddaEvenb(x, check=False):
                            if (flag_to_gen == 0 or flag_to_gen == 1):
                                posL3.append(x)
                        else:
                            if (flag_to_gen == 0 or flag_to_gen == 2):
                                negL3.append(x)

                    #lang:5
                    if lang == 0 or lang == 5:
                        if Lang_is_abBothEven(x, check=False):
                            if (flag_to_gen == 0 or flag_to_gen == 1):
                                posL5.append(x)
                        else:
                            if (flag_to_gen == 0 or flag_to_gen == 2):
                                negL5.append(x)

                    #lang:6
                    if lang == 0 or lang == 6:
                        if Lang_is_aMod3b(x, check=False):
                            if (flag_to_gen == 0 or flag_to_gen == 1):
                                posL6.append(x)
                        else:
                            if (flag_to_gen == 0 or flag_to_gen == 2):
                                negL6.append(x)

                    #lang:7
                    if lang == 0 or lang == 7:
                        if Lang_is_aStarbStaraStarbStar(x, check=False):
                            if (flag_to_gen == 0 or flag_to_gen == 1):
                                posL7.append(x)
                        else:
                            if (flag_to_gen == 0 or flag_to_gen == 2):
                                negL7.append(x)
            if break_now:
                break
        if break_now:
            break

    if lang == 0 or lang == 1:
        if Lang_is_aStar(emptyWord, check=False):
            posL1.append(emptyWord)
        else:
            negL1.append(emptyWord)

    if lang == 0 or lang == 2:
        if Lang_is_abSeq(emptyWord, check=False):
            posL2.append(emptyWord)
        else:
            negL2.append(emptyWord)

    if lang == 0 or lang == 3:
        if Lang_is_abSeq_OddaEvenb(emptyWord, check=False):
            posL3.append(emptyWord)
        else:
            negL3.append(emptyWord)

    if lang == 0 or lang == 5:
        if Lang_is_abBothEven(emptyWord, check=False):
            posL5.append(emptyWord)
        else:
            negL5.append(emptyWord)

    if lang == 0 or lang == 6:
        if Lang_is_aMod3b(emptyWord, check=False):
            posL6.append(emptyWord)
        else:
            negL6.append(emptyWord)

    if lang == 0 or lang == 7:
        if Lang_is_aStarbStaraStarbStar(emptyWord, check=False):
            posL7.append(emptyWord)
        else:
            negL7.append(emptyWord)

    return [posL1, negL1, posL2, negL2, posL3, negL3, posL5, negL5, posL6, negL6, posL7, negL7]

def generateExamplesByLang(lang: int, num_examples: int, test=False) -> None:
    if num_examples % 2 == 1:
        num_examples += 1
    num_examples_div2 = int(num_examples/2)
    #50% data is generated as per combination from 1 to 20.
    pos_neg_lang = generateExampleFromLengthOne(num_examples_div2, lang, flag_to_gen=2)
    if lang == 1:
        pos_idx, neg_idx = 0, 1
    elif lang == 2:
        pos_idx, neg_idx = 2, 3
    elif lang == 3:
        pos_idx, neg_idx = 4, 5
    posL, negL = pos_neg_lang[pos_idx], pos_neg_lang[neg_idx]
    print(f"Info: level 1, Number of examples for lang:{lang}, pos is {len(posL)}")
    print(f"Info: level 1, Number of examples for lang:{lang}, neg is {len(negL)}")

    posL += generatelimitedExamples(num_examples_div2 - len(posL), pos=True, lang=lang, maxlength=maxlength)
    negL += generatelimitedExamples(num_examples_div2 - len(negL), pos=False, lang=lang, maxlength=maxlength)

    print(f"Total number of examples for lang:{lang}, pos is {len(posL)}")
    print(f"Total number of examples for lang:{lang}, neg is {len(negL)}")


    # for i in range(0, 3):
    #     file_name_for_savingdata = "../dataOnQ/lang" + str(p[i])
    #     if i % 2 == 0:
    #         print(f"Info: writing... the length of lang {p[i]} pos examples: {len(posL)}")
    #         with open(file_name_for_savingdata + "_pos", "w+") as f:
    #             print(f"writing pos in {file_name_for_savingdata}")
    #             f.write(str(posL))
    #     else:
    #         print(f"Info: writing... the length of lang {p[i]} neg examples: {len(negL)}")
    #         with open(file_name_for_savingdata + "_neg", "w+") as f:
    #             print(f"writing pos in {file_name_for_savingdata}")
    #             f.write(str(negL))
    #write using pandas:
    padded_posL = [list(sublist) + [None]*(maxlength-len(sublist)) for sublist in posL]
    df_posL = pd.DataFrame(padded_posL, dtype='float64')
    padded_negL = [list(sublist) + [None] * (maxlength - len(sublist)) for sublist in negL]
    df_negL = pd.DataFrame(padded_negL, dtype='float64')
    file_name_for_savingdata = "../dataOnQ/lang" + str(lang)
    if test:
        file_name_for_savingdata = file_name_for_savingdata +  "_test"
    df_posL.to_csv(file_name_for_savingdata + "_pos", index=False)
    df_negL.to_csv(file_name_for_savingdata + "_neg", index=False)


# def generateExamples(num_examples: int) -> list:
#
#     pos_neg_lang = generateTypeExamples(lang=0, flag_to_gen=0)
#     first_level_count = [len(x) for x in pos_neg_lang]
#     maximum_num_ex = max(first_level_count)
#     p = [2, 2, 3, 3,  5, 5, 6, 6, 7, 7]
#     j = 0
#     for i in range(0, len(pos_neg_lang)):
#         if i%2 == 0:
#             print(f"Info: 1st level, the length of lang {p[i]} pos examples: {len(pos_neg_lang[i])}")
#         else:
#             print(f"Info: 1st level, the length of lang {p[i]} neg examples: {len(pos_neg_lang[i])}")
#
#
#     for lang, i in zip(p, range(len(first_level_count))):
#         if i%2 == 0:
#             pos = True
#         else:
#             pos = False
#         new_examples_list = generatelimitedExamples(maximum_num_ex-first_level_count[i], pos, lang, maxlength)
#         pos_neg_lang[i] = pos_neg_lang[i] + new_examples_list
#         print(f"Total number of examples for lang:{lang}, pos:{pos} is {len(pos_neg_lang[i])}")
#
#
#     for i in range(0, len(pos_neg_lang)):
#         file_name_for_savingdata = "lang" + p[j]
#         if i%2 == 0:
#             print(f"Info: 1st level, the length of lang {p[j]} pos examples: {len(pos_neg_lang[i])}")
#             with open("../../dataOnQ/" + file_name_for_savingdata + "_pos") as f:
#                 f.write(repr(pos_neg_lang[i]))
#         else:
#             print(f"Info: 1st level, the length of lang {p[j]} neg examples: {len(pos_neg_lang[i])}")
#             j += 1
#             with open("../../dataOnQ/" + file_name_for_savingdata + "_neg") as f:
#                 f.write(repr(pos_neg_lang[i]))

if __name__ == "__main__":
    # num_examples = 300000
    # generateExamples(num_examples)
    num_examples = 500000
    generateExamplesByLang(lang, num_examples)
    num_examples = 50000
    generateExamplesByLang(lang, num_examples, test=True)