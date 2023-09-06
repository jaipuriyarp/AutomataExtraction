# Language accepted is: (a)^*.
import sys
import os
import random
import ast

modelDir = "../"
pathsToInclude = ["../../TheoryOfEquality/vLStarForRationalAutomata/", modelDir, "../RNNLanguageModels", "."]
for path in pathsToInclude:
    sys.path.append(path)

from vLStar import RationalNumber, RationalNominalAutomata, learn

from rnnInterface import RNNInterface
from GTComparison import GTComparison
from groundTruthFunctions import Lang_is_aStar
from checkEquivalence import CheckEquivalence
from recordTime import RecordTime

# RNNModelName = "modelRNNQ_lang1_aStar.pt"
RNNModelName = "modelRNNQ_lang1_try_withlangGenOnQ.pt"
filelist = ["../dataOnQ/ang1_try_withlangGenOn_posL", "../dataOnQ/ang1_try_withlangGenOn_negL"]
RNNModelPath = os.path.join(modelDir, "models", RNNModelName)


rnnInterface = RNNInterface(rnn_model_path=RNNModelPath, input_size=1)
gTComparison = GTComparison(Lang_is_aStar)
checkEquivalence = CheckEquivalence(depth=7, num_of_RationalNumber=2,
                                    automaton=None, membershipQuery=None)
timer = RecordTime(record_elapsed_time=True)

def membershipQuery(word: list, printing=True) -> bool:
    # expects a list of RationalNumbers
    print(word)
    if len(word) and type(word[0]) != type(RationalNumber(None, None)):
        raise Exception("membershipQuery was called with the list: " + str(word) + "\n of type: " + str(type(word)))
    # print(f"The query is: {word}")
    rnnReply = rnnInterface.askRNN(word)
    Qreply = gTComparison.getGT(word, rnnReply, printing)
    word = rnnInterface.getRNNCompatibleInputFromRationalNumber(word, paranthesesLang=False)

    if (rnnReply != Qreply):
        if printing:
            print(f"membershipQuery: FOUND MISMATCH FOR {word}, rnn: {rnnReply} and GT: {Qreply}")
            timer.pause()
        else:
            print(f"equivalenceQuery: FOUND MISMATCH FOR {word}, rnn: {rnnReply} and GT: {Qreply}")

    # print (Qreply)
    if rnnReply:
        if printing:
            print(f"membershipQuery: {word} is in the language.")
    else:
        if printing:
            print(f"membershipQuery: {word} is not in the language.")
    # if printing:
    #     return rnnReply
    # return Qreply
    return rnnReply

def match_with_training_data(automaton: RationalNominalAutomata):
    numberOfExamples = 3000
    print(f"INFO: checking equivalence against Training data")
    # percent_of_data = 1
    pos, neg = convert_to_list_from_list(filelist=filelist)
    select_pos = random.sample(pos, int(numberOfExamples/2))
    select_neg = random.sample(neg, numberOfExamples - int(numberOfExamples/2))
    selected_data = select_pos + select_neg
    # num_elements_to_select = int(len(res) * (percent_of_data/100))
    # selected_data = random.sample(res, numberOfExamples)
    res_f = sorted(selected_data, key=len)

    for example in res_f:
        # if example in pos:
        #     isMember = True
        # else:
        #     isMember = False

        word = [RationalNumber(i, 1) for i in example]
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
def statisticalEquivalenceQuery(automaton: RationalNominalAutomata) -> tuple:
    numberOfExamples = 100
    print("Checking equivalence of the following automaton:")
    print(automaton)
    # checkEquivalence.setAutomaton(automaton)

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


    # for l in range(numberOfExamples):  # + numberOfEquivalenceQueriesAsked):
    #     length = 1 + int(l / 20)
    #     numerators = [random.randint(1, 5) for i in range(length)]  # random.sample(range(0, 5), length)
    #     denominators = [1] * length  # random.sample(range(1, 10 + 2 * l), length)
    #     word = [RationalNumber(numerators[i], denominators[i]) for i in range(length)]
    #
    #     isMember = membershipQuery(word, False)
    #     hypothesisIsMember = automaton.accepts(word)
    #
    #     if isMember != hypothesisIsMember:
    #         print("The languages are not equivalent, a counterexample is: " + str(word))
    #         if isMember:
    #             print("The word was rejected, but is in the language.")
    #         else:
    #             print("The word was accepted, but is not in the language.")
    #         return (False, word)
    #     else:
    #         if isMember:
    #             print(str(word) + " was correctly accepted")
    #         else:
    #             print(str(word) + " was correctly rejected")
    #
    # # print("The languages appear to be  equivalent after checking " + str(numberOfExamples) + " random examples")
    # print(f"The languages appear to be equivalent after checking  "
    #       f"{checkEquivalence.getQueriesCount() + numberOfExamples} random examples.")
    # timer.stop()
    # timer.report()
    # return (True, None)
    return match_with_training_data(automaton)

def convert_to_list_from_list(filelist=filelist) -> list:
    with open(filelist[0]) as f:
        pos = ast.literal_eval(f.readline())
    with open(filelist[1]) as f:
        neg = ast.literal_eval(f.readline())

    return [pos, neg]

def main() -> None:
    timer.start()
    pos_neg_list = convert_to_list_from_list(filelist=filelist)
    learnedAutomaton = learn(membershipQuery, statisticalEquivalenceQuery,
                             verbose=False, fileList=pos_neg_list)

    # checkEquivalence.setMembershipQuery(Lang_is_aStar)
    print(learnedAutomaton)
    gTComparison.statistics()


if __name__ == "__main__":
    main()
