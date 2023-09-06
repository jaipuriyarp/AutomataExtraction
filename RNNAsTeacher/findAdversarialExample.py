import sys
import os
import random
import ast
modelDir = "../models"
pathsToInclude = ["../../TheoryOfEquality/vLStarForRationalAutomata/", modelDir, "../RNNLanguageModels", "."]
for path in pathsToInclude:
    sys.path.append(path)

from vLStar import RationalNumber, RationalNominalAutomata, learn
from rnnInterface import RNNInterface
from GTComparison import GTComparison
from groundTruthFunctions import *
from recordTime import RecordTime
from EquivQueryGenerator import queryGenerator
filelist = None

#maimum run time is set to 5 hours
MAX_RUN_TIME = 18000 # in seconds
Upper_time_limit = 600 # in seconds
Upper_limit_of_sequence_checking = 5000
lang = 1
max_length = 20

if lang == 1:
    checkGndLabel = Lang_is_aStar
    model_name = "modelRNNQ_lang1_try_withlangGenOnQ.pt"
elif lang == 2:
    checkGndLabel = Lang_is_abSeq
    model_name = "modelRNNQ_lang2_try_withlangGenOnQ.pt"
elif lang == 3:
    checkGndLabel = Lang_is_abSeq_OddaEvenb
    model_name = "modelRNNQ_lang3_aOddbEvenNum_new.pt"
elif lang == 4:
    checkGndLabel = Lang_is_noTrigrams
    model_name = "modelRNNQ_lang4_try_withlangGenOnQ.pt"
elif lang == 5:
    checkGndLabel = Lang_is_abBothEven
    model_name = "modelRNNQ_lang5_try_withlangGenOnQ1.pt"
elif lang == 6:
    checkGndLabel = Lang_is_aMod3b
    model_name = "modelRNNQ_lang6_try_withlangGenOnQ.pt"
elif lang == 7:
    checkGndLabel = Lang_is_aStarbStaraStarbStar
    model_name = "modelRNNQ_lang7_try_withlangGenOnQ.pt"

RNNModelPath = os.path.join(modelDir, model_name)
rnnInterface = RNNInterface(rnn_model_path=RNNModelPath, input_size=1)
gTComparison = GTComparison(checkGndLabel)
# checkEquivalence = CheckEquivalence(depth=7, num_of_RationalNumber=2,
#                                     automaton=None, membershipQuery=None)
timer = RecordTime(record_elapsed_time=True)


print(f"Info: Lang: {lang}, model name: {model_name}, gnd function {checkGndLabel}")

def membershipQuery(word: list, printing=True) -> bool:
    # expects a list of RationalNumbers
    # print(word)
    if len(word) and type(word[0]) != type(RationalNumber(None, None)):
        raise Exception("membershipQuery was called with the list: " + str(word) + "\n of type: " + str(type(word)))
    # print(f"The query is: {word}")
    rnnReply = rnnInterface.askRNN(word)
    Qreply = gTComparison.getGT(word, rnnReply, printing)
    word = rnnInterface.getRNNCompatibleInputFromRationalNumber(word, paranthesesLang=False)

    if timer.getTotalRunTimeTillNow() >= MAX_RUN_TIME:
        print(f"WARNING: Exiting now. The MAX_RUN_TIME: {MAX_RUN_TIME} has been reached.")
        return None

    if gTComparison.queriesCount() >= Upper_limit_of_sequence_checking and \
            timer.getTotalRunTimeTillNow() >= Upper_time_limit:
        print(f"WARNING: The Upper limit of queries count: {Upper_limit_of_sequence_checking} and"
              f"the upper limit of run time: {Upper_time_limit} has been reached.")
        return None

    if (rnnReply != Qreply):
        if printing:
            print(f"membershipQuery: FOUND MISMATCH FOR {word}, rnn: {rnnReply} and GT: {Qreply}")
        else:
            print(f"equivalenceQuery: FOUND MISMATCH FOR {word}, rnn: {rnnReply} and GT: {Qreply}")
        time_elapsed = timer.pause()
        gTComparison.save_elapsed_time_for_query(word, time_elapsed)

    if rnnReply:
        if printing:
            print(f"membershipQuery: {word} is in the language.")
    else:
        if printing:
            print(f"membershipQuery: {word} is not in the language.")

    return rnnReply


def statisticalEquivalenceQuery(automaton: RationalNominalAutomata) -> tuple:
    print("Checking equivalence of the following automaton:")
    print(automaton)

    for eachSequence in queryGenerator(upper_limit_of_sequence_generation=Upper_limit_of_sequence_checking,
            max_length=max_length,current_query_count=gTComparison.queriesCount()):
        rationalNumberSequence = [RationalNumber(i,1) for i in eachSequence]
        isMember = membershipQuery(rationalNumberSequence, False)

        if isMember is None:
            timer.stop()
            timer.report()
            print(f"Final hypothesis is: {automaton}")
            return True, None

        hypothesisIsMember = automaton.accepts(rationalNumberSequence)

        if isMember != hypothesisIsMember:
            print(f"The languages are not equivalent, a counterexample is: {rationalNumberSequence}, length: {len(rationalNumberSequence)}")
            if isMember:
                print(f"The word was rejected, but is in the language.")
            else:
                print(f"The word was accepted, but is not in the language.")
            timer.pause()
            return (False, rationalNumberSequence)
        else:
            if isMember:
                print(f"{rationalNumberSequence} was correctly accepted")
            else:
                print(f"{rationalNumberSequence} was correctly rejected")

    timer.stop()
    # print(f"The final automata extracted is : {automaton}")
    print(f"The equivalence Query is passed and total number of queries count is {gTComparison.queriesCount()}")
    return True, None


def convert_to_list_from_list(filelist=filelist) -> list:
    with open(filelist[0]) as f:
        pos = ast.literal_eval(f.readline())
    with open(filelist[1]) as f:
        neg = ast.literal_eval(f.readline())
    return [pos, neg]

def main() -> None:
    numberOfExamples = 3000
    if filelist is not None:
        pos, neg = convert_to_list_from_list(filelist=filelist)
        select_pos = random.sample(pos, int(numberOfExamples / 2))
        select_neg = random.sample(neg, numberOfExamples - int(numberOfExamples / 2))
        selected_data = select_pos + select_neg
        global res_f
        res_f = sorted(selected_data, key=len)
        pos_neg_list = convert_to_list_from_list(filelist=filelist)
    else:
        pos_neg_list = []
        res_f = []


    timer.start()
    learnedAutomaton = learn(membershipQuery, statisticalEquivalenceQuery,
                             verbose=False, fileList=None)

    print(learnedAutomaton)
    timer.report()

    gTComparison.statistics(file_name= "lang" + str(lang) + "_list.csv")
    gTComparison.display_adversarial_query_time_relation(file_name= "lang" +str(lang) + "_adversarial_list.csv")


if __name__ == "__main__":
    main()
