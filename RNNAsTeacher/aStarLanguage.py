# Language accepted is: (a)^*.
import sys
import os
import random

# import torch

modelDir = "../"
pathsToInclude = ["../../TheoryOfEquality/vLStarForRationalAutomata/", modelDir, "../RNNLanguageModels"]
for path in pathsToInclude:
    sys.path.append(path)

from vLStar import RationalNumber, RationalNominalAutomata, learn

from rnnInterface import RNNInterface
from GTComparison import GTComparison
from groundTruthFunctions import Lang_is_aStar

RNNModelName = "modelRNN_lang1_aStar.pt"
RNNModelPath = os.path.join(modelDir, "models", RNNModelName)
word2vecModelPath = os.path.join(modelDir, "models", "GoogleNews-vectors-negative300.bin")

samples = [[RationalNumber(0, 1)],
           [RationalNumber(0, 1), RationalNumber(0, 1)],
           [RationalNumber(0, 1), RationalNumber(1, 1)],
           [RationalNumber(1, 1), RationalNumber(0, 1)],
           [RationalNumber(1, 1), RationalNumber(1, 1)],
           [RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(0, 1)],
           [RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(1, 1)],
           [RationalNumber(0, 1), RationalNumber(1, 1), RationalNumber(0, 1)],
           [RationalNumber(0, 1), RationalNumber(1, 1), RationalNumber(1, 1)],
           [RationalNumber(1, 1), RationalNumber(0, 1), RationalNumber(0, 1)],
           [RationalNumber(1, 1), RationalNumber(0, 1), RationalNumber(1, 1)],
           [RationalNumber(1, 1), RationalNumber(1, 1), RationalNumber(0, 1)],
           [RationalNumber(1, 1), RationalNumber(1, 1), RationalNumber(1, 1)],
           [RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(0, 1)],
           [RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(1, 1)],
           [RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(1, 1), RationalNumber(0, 1)],
           [RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(1, 1), RationalNumber(1, 1)],
           [RationalNumber(0, 1), RationalNumber(1, 1), RationalNumber(0, 1), RationalNumber(0, 1)],
           [RationalNumber(0, 1), RationalNumber(1, 1), RationalNumber(0, 1), RationalNumber(1, 1)],
           [RationalNumber(0, 1), RationalNumber(1, 1), RationalNumber(1, 1), RationalNumber(0, 1)],
           [RationalNumber(0, 1), RationalNumber(1, 1), RationalNumber(1, 1), RationalNumber(1, 1)],
           [RationalNumber(1, 1), RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(0, 1)],
           [RationalNumber(1, 1), RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(1, 1)],
           [RationalNumber(1, 1), RationalNumber(0, 1), RationalNumber(1, 1), RationalNumber(0, 1)],
           [RationalNumber(1, 1), RationalNumber(0, 1), RationalNumber(1, 1), RationalNumber(1, 1)],
           [RationalNumber(1, 1), RationalNumber(1, 1), RationalNumber(0, 1), RationalNumber(0, 1)],
           [RationalNumber(1, 1), RationalNumber(1, 1), RationalNumber(0, 1), RationalNumber(1, 1)],
           [RationalNumber(1, 1), RationalNumber(1, 1), RationalNumber(1, 1), RationalNumber(0, 1)],
           [RationalNumber(1, 1), RationalNumber(1, 1), RationalNumber(1, 1), RationalNumber(1, 1)],
           [RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(0, 1),
            RationalNumber(0, 1)],
           [RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(0, 1), RationalNumber(0, 1),
            RationalNumber(0, 1), RationalNumber(0, 1)]]

rnnInterface = RNNInterface(word2vec_model_path=word2vecModelPath, rnn_model_path=RNNModelPath)
gTComparison = GTComparison(Lang_is_aStar)


def membershipQuery(word: list, printing=True) -> bool:
    # expects a list of RationalNumbers
    if len(word) and type(word[0]) != type(RationalNumber(None, None)):
        raise Exception("membershipQuery was called with the list: " + str(word) + "\n of type: " + str(type(word)))
    print(f"The query is: {word}")
    rnnReply = rnnInterface.askRNN(word)
    Qreply = gTComparison.getGT(word, rnnReply, printing)
    # print (Qreply)
    if rnnReply:
        if printing:
            print(f"membershipQuery: {word} is in the language.")
    else:
        if printing:
            print(f"membershipQuery: {word} is not in the language.")
    return Qreply


def statisticalEquivalenceQuery(automaton: RationalNominalAutomata) -> tuple:
    global samples
    numberOfExamples = 100

    print("Checking equivalence of the following automaton:")
    print(automaton)

    for word in samples:
        isMember = membershipQuery(word, False)
        hypothesisIsMember = automaton.accepts(word)

        if isMember != hypothesisIsMember:
            print("The languages are not equivalent, a counterexample is: " + str(word))
            if isMember:
                print("The word was rejected, but is in the language.")
            else:
                print("The word was accepted, but is not in the language.")
            return (False, word)
        else:
            if isMember:
                print(str(word) + " was correctly accepted")
            else:
                print(str(word) + " was correctly rejected")

    for l in range(numberOfExamples):  # + numberOfEquivalenceQueriesAsked):
        length = 1 + int(l / 20)
        numerators = [random.randint(1, 5) for i in range(length)]  # random.sample(range(0, 5), length)
        denominators = [1] * length  # random.sample(range(1, 10 + 2 * l), length)
        word = [RationalNumber(numerators[i], denominators[i]) for i in range(length)]

        isMember = membershipQuery(word, False)
        hypothesisIsMember = automaton.accepts(word)

        if isMember != hypothesisIsMember:
            print("The languages are not equivalent, a counterexample is: " + str(word))
            if isMember:
                print("The word was rejected, but is in the language.")
            else:
                print("The word was accepted, but is not in the language.")
            return (False, word)
        else:
            if isMember:
                print(str(word) + " was correctly accepted")
            else:
                print(str(word) + " was correctly rejected")
    print("The languages appear to be  equivalent after checking " + str(numberOfExamples) + " random examples")
    return (True, None)


def main() -> None:
    learnedAutomaton = learn(membershipQuery, statisticalEquivalenceQuery)
    print(learnedAutomaton)
    gTComparison.statistics()


if __name__ == "__main__":
    main()
