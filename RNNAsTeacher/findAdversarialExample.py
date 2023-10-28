import sys
import os
import random
import ast
import argparse
modelDir = "../models"
pathsToInclude = ["../../TheoryOfEquality/vLStarForRationalAutomata/", modelDir, "../RNNLanguageModels", "."]
for path in pathsToInclude:
    sys.path.append(path)

file_suffix_name = "_newX"
from vLStar import RationalNumber, RationalNominalAutomata, learn
from rnnInterface import RNNInterface
from GTComparison import GTComparison
from groundTruthFunctions import *
from recordTime import RecordTime
from EquivQueryGenerator import queryGenerator, generateOneQuery
filelist = None

#maimum run time is set to 5 hours
MAX_RUN_TIME = 18000 # in seconds
# Upper_time_limit = 600 # in seconds
Upper_time_limit = 400 # in seconds
Upper_limit_of_sequence_checking = 2000
# Upper_time_limit = 30 # in seconds
# Upper_limit_of_sequence_checking = 50
lang = 2
max_length = 20
# global count_function_calls
count_function_calls = 0
printQlimit, printTlimit = False, False
def get_gndFunc(lang):
    if lang == 1:
        checkGndLabel = Lang_is_aStar
        # model_name = "modelRNNQ_lang1_try_withlangGenOnQ.pt"
    elif lang == 2:
        checkGndLabel = Lang_is_abSeq
        # model_name = "modelRNNQ_lang2_try_withlangGenOnQ.pt"
        # model_name = "modelRNNQ_lang2_newX.pt"
    elif lang == 3:
        checkGndLabel = Lang_is_abSeq_OddaEvenb
        # model_name = "modelRNNQ_lang3_aOddbEvenNum_new.pt"
        # model_name = "modelRNNQ_lang3_newX.pt"
    elif lang == 4:
        checkGndLabel = Lang_is_noTrigrams
        # model_name = "modelRNNQ_lang4_try_withlangGenOnQ.pt"
        # model_name = "modelRNNQ_lang2_newX.pt"
    elif lang == 5:
        checkGndLabel = Lang_is_abBothEven
        # model_name = "modelRNNQ_lang5_try_withlangGenOnQ1.pt"
    elif lang == 6:
        checkGndLabel = Lang_is_aMod3b
        # model_name = "modelRNNQ_lang6_try_withlangGenOnQ.pt"
    elif lang == 7:
        checkGndLabel = Lang_is_aStarbStaraStarbStar
        # model_name = "modelRNNQ_lang7_try_withlangGenOnQ.pt"
    else:
        raise Exception(f"No such languages!!")

    return checkGndLabel

def membershipQuery(word: list, printing=True) -> bool:
    # expects a list of RationalNumbers
    # print(word)
    if len(word) and type(word[0]) != type(RationalNumber(None, None)):
        raise Exception("membershipQuery was called with the list: " + str(word) + "\n of type: " + str(type(word)))

    if timer.getTotalRunTimeTillNow() >= MAX_RUN_TIME:
        print(f"WARNING: Exiting now. The MAX_RUN_TIME: {MAX_RUN_TIME} has been reached.")
        timer.stop()
        return None

    global printQlimit, printTlimit

    if gTComparison.queriesCount() >= Upper_limit_of_sequence_checking and printQlimit == False:
        print(f"DIAMOND: No. of qCount: {gTComparison.queriesCount()} ",
              f"No. of memQ count: {gTComparison.num_pos_memQ + gTComparison.num_pos_memQ} ",
              f"No. of EquivQ count: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ} ",
              f"No. of adversarial memQ: {gTComparison.num_pos_memQ + gTComparison.num_neg_memQ - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ)} ",
              f"No. of adversarial EquivQ: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ - (gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
              f"No. of adversarial examples: {gTComparison.queriesCount() - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ + gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
              f"Time: {timer.getTotalRunTimeTillNow()}")
        printQlimit = True

    if timer.getTotalRunTimeTillNow() >= Upper_time_limit and printTlimit == False:
        print(f"DIAMOND: No. of qCount: {gTComparison.queriesCount()} ",
              f"No. of memQ count: {gTComparison.num_pos_memQ + gTComparison.num_pos_memQ} ",
              f"No. of EquivQ count: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ} ",
              f"No. of adversarial memQ: {gTComparison.num_pos_memQ + gTComparison.num_neg_memQ - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ)} ",
              f"No. of adversarial EquivQ: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ - (gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
              f"No. of adversarial examples: {gTComparison.queriesCount() - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ + gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
              f"Time: {timer.getTotalRunTimeTillNow()}")
        printTlimit = True

    if gTComparison.queriesCount() >= Upper_limit_of_sequence_checking and \
            timer.getTotalRunTimeTillNow() >= Upper_time_limit:
        print(f"WARNING: The Upper limit of queries count: {Upper_limit_of_sequence_checking} and "
              f"the upper limit of run time: {Upper_time_limit} has been reached.")
        timer.stop()
        return None

    rnnReply = rnnInterface.askRNN(word)
    Qreply = gTComparison.getGT(word, rnnReply, printing)
    word = rnnInterface.getRNNCompatibleInputFromRationalNumber(word, paranthesesLang=False)

    if (rnnReply != Qreply):
        if printing:
            print(f"membershipQuery: FOUND MISMATCH FOR {word}, rnn: {rnnReply} and GT: {Qreply}")
        else:
            print(f"equivalenceQuery: FOUND MISMATCH FOR {word}, rnn: {rnnReply} and GT: {Qreply}")
        time_elapsed = timer.pause()
        gTComparison.save_elapsed_time_for_query(word, time_elapsed, printing)

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

    # for eachSequence in queryGenerator(upper_limit_of_sequence_generation=Upper_limit_of_sequence_checking,
    #         max_length=max_length,current_query_count=gTComparison.queriesCount()):
    global count_function_calls
    global printQlimit, printTlimit
    while timer.getTotalRunTimeTillNow() < Upper_time_limit or \
            gTComparison.queriesCount() < Upper_limit_of_sequence_checking:

        if gTComparison.queriesCount() >= Upper_limit_of_sequence_checking and printQlimit == False:
            print(f"DIAMOND: No. of qCount: {gTComparison.queriesCount()} ",
                  f"No. of memQ count: {gTComparison.num_pos_memQ + gTComparison.num_pos_memQ} ",
                  f"No. of EquivQ count: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ} ",
                  f"No. of adversarial memQ: {gTComparison.num_pos_memQ + gTComparison.num_neg_memQ - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ)} ",
                  f"No. of adversarial EquivQ: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ - (gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
                  f"No. of adversarial examples: {gTComparison.queriesCount() - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ + gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
                  f"Time: {timer.getTotalRunTimeTillNow()}")
            printQlimit = True

        if timer.getTotalRunTimeTillNow() >= Upper_time_limit and printTlimit == False:
            print(f"DIAMOND: No. of qCount: {gTComparison.queriesCount()} ",
                  f"No. of memQ count: {gTComparison.num_pos_memQ + gTComparison.num_pos_memQ} ",
                  f"No. of EquivQ count: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ} ",
                  f"No. of adversarial memQ: {gTComparison.num_pos_memQ + gTComparison.num_neg_memQ - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ)} ",
                  f"No. of adversarial EquivQ: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ - (gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
                  f"No. of adversarial examples: {gTComparison.queriesCount() - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ + gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
                  f"Time: {timer.getTotalRunTimeTillNow()}")
            printTlimit = True

        for sequence in generateOneQuery(max_length, no_of_times_called=count_function_calls):
            count_function_calls += 1
            rationalNumberSequence = [RationalNumber(i,1) for i in sequence]
            isMember = membershipQuery(rationalNumberSequence, False)

            if isMember is None:
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

    if gTComparison.queriesCount() >= Upper_limit_of_sequence_checking and printQlimit == False:
        print(f"DIAMOND: No. of qCount: {gTComparison.queriesCount()} ",
              f"No. of memQ count: {gTComparison.num_pos_memQ + gTComparison.num_pos_memQ} ",
              f"No. of EquivQ count: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ} ",
              f"No. of adversarial memQ: {gTComparison.num_pos_memQ + gTComparison.num_neg_memQ - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ)} ",
              f"No. of adversarial EquivQ: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ - (gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
              f"No. of adversarial examples: {gTComparison.queriesCount() - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ + gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
              f"Time: {timer.getTotalRunTimeTillNow()}")
        printQlimit = True

    if timer.getTotalRunTimeTillNow() >= Upper_time_limit and printTlimit == False:
        print(f"DIAMOND: No. of qCount: {gTComparison.queriesCount()} ",
              f"No. of memQ count: {gTComparison.num_pos_memQ + gTComparison.num_pos_memQ} ",
              f"No. of EquivQ count: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ} ",
              f"No. of adversarial memQ: {gTComparison.num_pos_memQ + gTComparison.num_neg_memQ - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ)} ",
              f"No. of adversarial EquivQ: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ - (gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
              f"No. of adversarial examples: {gTComparison.queriesCount() - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ + gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
              f"Time: {timer.getTotalRunTimeTillNow()}")
        printTlimit = True


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
    parser = argparse.ArgumentParser(description='Process some language options.')
    parser.add_argument('--lang', help='Specify the programming language.')
    args = parser.parse_args()
    lang = args.lang
    lang = int(lang)

    checkGndLabel = get_gndFunc(lang)
    model_name = "modelRNNQ_lang" + str(lang) + "_newX.pt"
    RNNModelPath = os.path.join(modelDir, model_name)
    global rnnInterface,  gTComparison, timer
    rnnInterface = RNNInterface(rnn_model_path=RNNModelPath, input_size=1)
    gTComparison = GTComparison(checkGndLabel)
    timer = RecordTime(record_elapsed_time=False)

    print(f"Info: Lang: {lang}, model name: {model_name}, gnd function {checkGndLabel}")
    global res_f

    if filelist is not None:
        numberOfExamples = 3000
        pos, neg = convert_to_list_from_list(filelist=filelist)
        select_pos = random.sample(pos, int(numberOfExamples / 2))
        select_neg = random.sample(neg, numberOfExamples - int(numberOfExamples / 2))
        selected_data = select_pos + select_neg
        res_f = sorted(selected_data, key=len)
        pos_neg_list = convert_to_list_from_list(filelist=filelist)
    else:
        pos_neg_list = []
        res_f = []


    timer.start()
    learnedAutomaton = learn(membershipQuery, statisticalEquivalenceQuery,
                             verbose=False, fileList=None)
    print(learnedAutomaton)
    # timer.report()
    pt_file = "../logs/" + "lang" + str(lang) + "_presentationTable" + file_suffix_name + ".csv"
    gTComparison.create_presentation_table(file_name=pt_file)
    st_file = "../logs/" +  "lang" + str(lang) + "_list" + file_suffix_name + ".csv"
    gTComparison.statistics(file_name=st_file)
    comp_file = "../logs/" + "lang" + str(lang) + "_adversarial_list" + file_suffix_name + ".csv"
    gTComparison.display_adversarial_query_time_relation(file_name=comp_file)


if __name__ == "__main__":
    main()

