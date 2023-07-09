# Language accepted is: (ab)^n, where a<b.
import sys
import os
import random

# import torch

modelDir = "../"
pathsToInclude = ["../../vLStarForRationalAutomata/", modelDir, "../RNNLanguageModels"]
for path in pathsToInclude:
    sys.path.append(path)

from vLStar import RationalNumber, RationalNominalAutomata, learn

from rnnInterface import RNNInterface
from GTComparison import GTComparison
from groundTruthFunctions import Lang_is_abSeq_aLessThanb

RNNModelName = "modelRNNQ_lang2_abSeq_aLessThanb.pt"
RNNModelPath = os.path.join(modelDir, "models", RNNModelName)

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

rnnInterface = RNNInterface(rnn_model_path=RNNModelPath, input_size=1)
gTComparison = GTComparison(Lang_is_abSeq_aLessThanb)


def membershipQuery(word: list, printing=True) -> bool:
    # expects a list of RationalNumbers
    if len(word) and type(word[0]) != type(RationalNumber(None, None)):
        raise Exception("membershipQuery was called with the list: " + str(word) + "\n of type: " + str(type(word)))
    print(f"The query is: {word}")
    rnnReply = rnnInterface.askRNN(word)
    Qreply = gTComparison.getGT(word, rnnReply, printing)

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
            return word
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
            return word
        else:
            if isMember:
                print(str(word) + " was correctly accepted")
            else:
                print(str(word) + " was correctly rejected")
    print("The languages appear to be  equivalent after checking " + str(numberOfExamples) + " random examples")
    return None


def main() -> None:
    learnedAutomaton = learn(membershipQuery, statisticalEquivalenceQuery)
    print(learnedAutomaton)
    gTComparison.statistics()


if __name__ == "__main__":
    main()
