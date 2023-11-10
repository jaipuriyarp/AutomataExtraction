# Language accepted is: balanced parentheses.
# ():
# ( : 10 : 2
# ) : 01 : 1
import sys
import os
import argparse
import random

logs_dir = "../logs"
model_dir = "../models"
pathsToInclude = ["../../vLStarForRationalAutomata/", model_dir, "../RNNLanguageModels"]
for path in pathsToInclude:
    sys.path.append(path)

from vLStar import RationalNumber, RationalNominalAutomata, learn
from rnnInterface import RNNInterface
from GTComparison import GTComparison
from groundTruthFunctions import is_balanced_parenthesis as checkGndLabel
from recordTime import RecordTime
from EquivQueryGenerator import queryGenerator, generateOneQuery


#maimum run time is set to 5 hours
MAX_RUN_TIME = 18000 # in seconds
# Upper_time_limit = 600 # in seconds
Upper_time_limit = 400 # in seconds
Upper_limit_of_sequence_checking = 2000
printQlimit, printTlimit = False, False

# model_name = "modelRNNQ_lang8_balancedParenthesis.pt"
# model_name = "modelRNNQ_lang8_balancedParenthesis_allEx1.pt"
# model_name = "modelRNNQ_lang8_balancedParenthesis_includingForeignelts.pt"
# model_name  = "modelRNNQ_lang8_balancedParenthesis_allEx1.pt"

count_function_calls = 0


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

    rnnReply = rnnInterface.askRNN(word, paranthesesLang=(lang == 8))
    word = rnnInterface.getRNNCompatibleInputFromRationalNumber(word, paranthesesLang=(lang == 8))
    Qreply = gTComparison.getGT(word, rnnReply, printing)


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
# def membershipQuery(word: list, printing=True) -> bool:
#     # expects a list of RationalNumbers
#     if len(word) and type(word[0]) != type(RationalNumber(0,1)) :
#         raise Exception("membershipQuery was called with the list: " + str(word) + "\n of type: " + str(type(word)))
#
#     if len(word) > max_length:
#         print(f"WARNING: Exiting now. The RNN maximum length: {max_length} limit has been reached.")
#         timer.stop()
#         return None
#
#     if timer.getTotalRunTimeTillNow() >= MAX_RUN_TIME:
#         print(f"WARNING: Exiting now. The MAX_RUN_TIME: {MAX_RUN_TIME} has been reached.")
#         timer.stop()
#         return None
#
#     if gTComparison.queriesCount() >= Upper_limit_of_sequence_checking and \
#             timer.getTotalRunTimeTillNow() >= Upper_time_limit:
#         print(f"WARNING: The Upper limit of queries count: {Upper_limit_of_sequence_checking} and "
#               f"the upper limit of run time: {Upper_time_limit} has been reached.")
#         timer.stop()
#         return None
#
#     rnnReply = rnnInterface.askRNN(word, paranthesesLang=(lang == 8))
#     word = rnnInterface.getRNNCompatibleInputFromRationalNumber(word, paranthesesLang=(lang == 8))
#     Qreply = gTComparison.getGT(word, rnnReply, printing)
#
#     if (rnnReply != Qreply):
#         if printing:
#             print(f"membershipQuery: FOUND MISMATCH FOR {word}, rnn: {rnnReply} and GT: {Qreply}")
#         else:
#             print(f"equivalenceQuery: FOUND MISMATCH FOR {word}, rnn: {rnnReply} and GT: {Qreply}")
#         time_elapsed = timer.pause()
#         gTComparison.save_elapsed_time_for_query(word, time_elapsed, printing)
#
#     if rnnReply:
#         if printing:
#             print(f"membershipQuery: {word} is in the language.")
#     else:
#         if printing:
#             print(f"membershipQuery: {word} is not in the language.")
#
#     return rnnReply


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

        for sequence in generateOneQuery(max_length, no_of_times_called=count_function_calls, lang=8):
            count_function_calls += 1
            rationalNumberSequence = [RationalNumber(1, 1) if char == '(' else RationalNumber(0, 1) for char in sequence]
            print(f"sequence: {sequence}, rational: {rationalNumberSequence}")
            isMember = membershipQuery(rationalNumberSequence, False)

            if isMember is None:
                timer.report()
                print(f"Final hypothesis is: {automaton}")
                return None

            hypothesisIsMember = automaton.accepts(rationalNumberSequence)

            if isMember != hypothesisIsMember:
                print(f"The languages are not equivalent, a counterexample is: {rationalNumberSequence}, length: {len(rationalNumberSequence)}")
                if isMember:
                    print(f"The word was rejected, but is in the language.")
                else:
                    print(f"The word was accepted, but is not in the language.")
                timer.pause()
                return rationalNumberSequence
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
    return None

def main() -> None:
    parser = argparse.ArgumentParser(
        description='This file is used for creating a balanced dataset for language 8 (i.e. balanced parentheses language).')
    parser.add_argument('--file_suffix', type=str, default="",
                        help='Specify the suffix of the file for saving the data.\n Default value is None.')
    parser.add_argument('--model_suffix', type=str, default="",
                        help='Specify the suffix of the RNN model name.\n Default value is None.')
    args = parser.parse_args()
    file_suffix = args.file_suffix
    model_suffix = args.model_suffix

    global lang, max_length
    lang, max_length = 8, 24

    # checkGndLabel = is_balanced_parenthesis
    model_name = "modelRNNQ_lang" + str(lang)
    if model_suffix != "":
        model_name = model_name + "_" + model_suffix
    else:
        model_name = model_name + ".pt"

    RNNModelPath = os.path.join(model_dir, model_name)
    global rnnInterface, gTComparison, timer
    rnnInterface = RNNInterface(rnn_model_path=RNNModelPath, input_size=2, maxlength=max_length)
    gTComparison = GTComparison(checkGndLabel)
    timer = RecordTime(record_elapsed_time=False)

    print(f"Info: Lang: {lang}, model name: {model_name}, gnd function {checkGndLabel}")

    timer.start()
    learnedAutomaton = learn(membershipQuery, statisticalEquivalenceQuery, paranthesesLang=True)
    print(learnedAutomaton)
    print(f"DIAMOND_final: No. of qCount: {gTComparison.queriesCount()} ",
          f"No. of memQ count: {gTComparison.num_pos_memQ + gTComparison.num_pos_memQ} ",
          f"No. of EquivQ count: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ} ",
          f"No. of adversarial memQ: {gTComparison.num_pos_memQ + gTComparison.num_neg_memQ - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ)} ",
          f"No. of adversarial EquivQ: {gTComparison.num_pos_EquivQ + gTComparison.num_neg_EquivQ - (gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
          f"No. of adversarial examples: {gTComparison.queriesCount() - (gTComparison.true_positive_memQ + gTComparison.true_negative_memQ + gTComparison.true_positive_EquivQ + gTComparison.true_negative_EquivQ)} ",
          f"Time: {timer.getTotalRunTimeTillNow()}")
    # timer.report()
    # gTComparison.create_presentation_table(file_name="lang" + str(lang) + "_presentationTable" + file_suffix + ".csv")
    # gTComparison.statistics(file_name="lang" + str(lang) + "_list" + file_suffix + ".csv")
    # gTComparison.display_adversarial_query_time_relation(file_name="lang" + str(lang) + "_adversarial_list" + file_suffix + ".csv")

    pt_file = os.path.join(logs_dir, "lang" + str(lang) + "_presentationTable" + file_suffix + ".csv")
    gTComparison.create_presentation_table(file_name=pt_file)
    st_file = os.path.join(logs_dir, "lang" + str(lang) + "_list" + file_suffix + ".csv")
    gTComparison.statistics(file_name=st_file)
    comp_file = os.path.join(logs_dir, "lang" + str(lang) + "_adversarial_list" + file_suffix + ".csv")
    gTComparison.display_adversarial_query_time_relation(file_name=comp_file)

if __name__ == "__main__":
    main()
