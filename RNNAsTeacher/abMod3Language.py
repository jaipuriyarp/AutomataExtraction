'''
Language accepted is: any string without Trigram.
'''
import sys
import os
import random

modelDir = "../"
pathsToInclude = ["../../TheoryOfEquality/vLStarForRationalAutomata/", modelDir, "../RNNLanguageModels"]
for path in pathsToInclude:
    sys.path.append(path)

from vLStar import RationalNumber, RationalNominalAutomata, learn
from rnnInterface import RNNInterface
from GTComparison import GTComparison
from groundTruthFunctions import Lang_is_aMod3b
from checkEquivalence import CheckEquivalence
from recordTime import RecordTime

RNNModelName = "modelRNNQ_lang6_amod3b.pt"
RNNModelPath = os.path.join(modelDir, "models", RNNModelName)

rnnInterface = RNNInterface(rnn_model_path=RNNModelPath, input_size=1)
gTComparison = GTComparison(Lang_is_aMod3b)
checkEquivalence = CheckEquivalence(depth=8, num_of_RationalNumber=3,
                                    automaton=None, membershipQuery=None)
timer = RecordTime(record_elapsed_time=True)

def membershipQuery(word: list, printing=True) -> bool:
    # expects a list of RationalNumbers
    if len(word) and type(word[0]) != type(RationalNumber(None, None)):
        raise Exception("membershipQuery was called with the list: " + str(word) + "\n of type: " + str(type(word)))
    # print(f"The query is: {word}")

    rnnReply = bool(rnnInterface.askRNN(word)[-1])
    Qreply = gTComparison.getGT(word, rnnReply, printing)
    word = rnnInterface.getRNNCompatibleInputFromRationalNumber(word, paranthesesLang=False)

    # print (Qreply)
    if (rnnReply != Qreply):
        if printing:
            print(f"membershipQuery: FOUND MISMATCH FOR {word}, rnn: {rnnReply} and GT: {Qreply}")
            timer.stop()
            timer.reset()
            timer.start()
            if len(set(word)) > 2:
                rnnReply = False
                print(f"WARNING: RNN reply set to false since the word contained more than two unique values.")
        else:
            print(f"equivalenceQuery: FOUND MISMATCH FOR {word}, rnn: {rnnReply} and GT: {Qreply}")

    if Qreply:
        if printing:
            print(f"membershipQuery: {word} is in the language.")
    else:
        if printing:
            print(f"membershipQuery: {word} is not in the language.")
    if printing:
        return rnnReply
    return Qreply


def statisticalEquivalenceQuery(automaton: RationalNominalAutomata) -> tuple:
    global samples
    numberOfExamples = 100

    print("Checking equivalence of the following automaton:")
    print(automaton)

    for word in checkEquivalence.generateQueries():
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
    timer.stop()
    timer.report()
    return (True, None)


def main() -> None:
    timer.start()
    learnedAutomaton = learn(membershipQuery, statisticalEquivalenceQuery)
    print(learnedAutomaton)
    gTComparison.statistics()


if __name__ == "__main__":
    main()
