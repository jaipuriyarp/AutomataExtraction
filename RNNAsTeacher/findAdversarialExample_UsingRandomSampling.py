import sys
import os
import argparse
modelDir = "../models"
pathsToInclude = ["../../TheoryOfEquality/vLStarForRationalAutomata/", modelDir, "../RNNLanguageModels", "."]
for path in pathsToInclude:
    sys.path.append(path)

file_extension_name = "_newX"

from vLStar import RationalNumber, RationalNominalAutomata, learn
from rnnInterface import RNNInterface
from GTComparison import GTComparison
from groundTruthFunctions import *
from recordTime import RecordTime
from EquivQueryGenerator import queryGenerator, generateOneQuery
# global Upper_time_limit, Upper_limit_of_sequence_checking
#maimum run time is set to 5 hours
MAX_RUN_TIME = 18000 # in seconds
# Upper_time_limit = 600 # in seconds
Upper_time_limit_global = 400 # in seconds
Upper_limit_of_sequence_checking_global = 2000
# Upper_time_limit = 30 # in seconds
# Upper_limit_of_sequence_checking = 50
# lang = 4
count_function_calls = 0
printQlimit, printTlimit = False, False
def compare_rnn_with_gt(sequence: list, rnnInterface:object, gTComparison:object, timer:object):
    if lang == 8:
        rationalNumberSequence = sequence
    else:
        rationalNumberSequence = [RationalNumber(i, 1) for i in sequence]

    rnnReply = rnnInterface.askRNN(rationalNumberSequence, paranthesesLang=(lang == 8))
    groundTruth = gTComparison.getGT(rationalNumberSequence, rnnReply, True)
    # word = rnnInterface.getRNNCompatibleInputFromRationalNumber(rationalNumberSequence, paranthesesLang=(lang == 8))
    # print(f"word is {word}")

    if (rnnReply != groundTruth):
        print(f"Query: FOUND MISMATCH FOR {rationalNumberSequence}, rnn: {rnnReply} and GT: {groundTruth}")
        time_elapsed = timer.pause()
        gTComparison.save_elapsed_time_for_query(rationalNumberSequence, time_elapsed, True)

    else:
        if rnnReply:
            print(f"Query: {rationalNumberSequence} is in the language.")
        else:
            print(f"Query: {rationalNumberSequence} is not in the language.")

def findAdversarialExamples(max_length:int, rnnInterface:object, gTComparison:object, timer:object,
                            Upper_time_limit:int, Upper_limit_of_sequence_checking:int) -> None:

    global count_function_calls
    global printQlimit, printTlimit
    while gTComparison.queriesCount() < Upper_limit_of_sequence_checking or \
                          timer.getTotalRunTimeTillNow() < Upper_time_limit:

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

        for sequence in generateOneQuery(max_length, count_function_calls, lang):
            compare_rnn_with_gt(sequence, rnnInterface, gTComparison, timer)
            count_function_calls += 1

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


def main() -> None:
    parser = argparse.ArgumentParser(description='A Python script that accepts input using -lang.')
    parser.add_argument('--lang', type=str, help='Specify an input value.')
    args = parser.parse_args()
    global lang
    lang = args.lang

    print(f"INFO: Lang selected is : {lang}")
    input_size, max_length  = 1, 20
    model_name, checkGndLabel = None, None
    lang = int(lang)

    Upper_time_limit, Upper_limit_of_sequence_checking = Upper_time_limit_global, Upper_limit_of_sequence_checking_global
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
    elif lang == 8:
        input_size, max_length = 2, 24
        checkGndLabel = is_balanced_parenthesis
        model_name = "modelRNNQ_lang8_balancedParenthesis.pt"
        model_name = "modelRNNQ_lang8_balancedParenthesis_allEx1.pt"

    dir_to_save = "../logs/"
    model_name = "modelRNNQ_lang" + str(lang) + "_newX.pt"
    print(f"Info: Lang: {lang}, model name: {model_name}, gnd function: {checkGndLabel}")
    RNNModelPath = os.path.join(modelDir, model_name)
    rnnInterface = RNNInterface(rnn_model_path=RNNModelPath, input_size=input_size, maxlength=max_length)
    gTComparison = GTComparison(checkGndLabel)
    # checkEquivalence = CheckEquivalence(depth=7, num_of_RationalNumber=2,
    #                                     automaton=None, membershipQuery=None)
    timer = RecordTime(record_elapsed_time=False)



    timer.start()
    findAdversarialExamples(max_length=max_length, rnnInterface=rnnInterface, gTComparison=gTComparison,
                                timer=timer, Upper_time_limit=Upper_time_limit, Upper_limit_of_sequence_checking=Upper_limit_of_sequence_checking)

    # timer.report()

    gTComparison.create_presentation_table(file_name= dir_to_save + "lang" + str(lang) + "_presentationTable_rSampling" + file_extension_name + ".csv")
    gTComparison.statistics(file_name= dir_to_save + "lang" + str(lang) + "_list_rSampling" + file_extension_name + ".csv")
    gTComparison.display_adversarial_query_time_relation(file_name=  dir_to_save + "lang" + str(lang) + "_adversarial_list_rSampling" + file_extension_name + ".csv")


if __name__ == "__main__":
    main()

