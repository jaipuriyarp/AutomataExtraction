# Language accepted is: balanced parentheses.
import sys
import os
import random
import torch
import pandas as pd
from fractions import Fraction

modelDir = "../"
pathsToInclude = ["../../vLStarForRationalAutomata/", modelDir, "../RNNLanguageModels"]
for path in pathsToInclude:
    sys.path.append(path)

from vLStar import RationalNumber, RationalNominalAutomata, learn


# from GTComparison import GTComparison
from torch.utils.data import DataLoader
from timeSeriesDataSet import TimeSeriesDataSet, MyDataSet
from trainOnTimeSeriesData import test_model
from rnnModel import RNNModel


RNNModelName = "modelRNN_timeSeries_acc82.5_seqlen_128.pt"
RNNModelPath = os.path.join(modelDir, "models", RNNModelName)
# gTComparison = GTComparison(is_balanced_parenthesis)
input_size = 4
hidden_size = 256
output_size = 1
num_layers = 2
rnn_model = RNNModel(input_size=input_size, hidden_size=hidden_size,
                     output_size=output_size, num_layers=num_layers, model_name=RNNModelName)
rnn_model.load_RNN_model(RNNModelPath)

def convertRationalNumberListToFloat(rationalNumList: list) -> list:
    '''This function converts Rational number to floating point number'''
    return [num.getNumerator() / num.getDenominator() for num in rationalNumList]

def convertFloatToRationalNumberList(floatNumList: list) -> list:
    res = []
    for number in floatNumList:
        fraction = Fraction(number).limit_denominator()
        res.append(RationalNumber(fraction.numerator, fraction.denominator))
    return res

    # return
def convertRationalNumToRNNCompatible(word:list):
    '''This function gets a list of Rational number list and converts the query to RNN compatible'''
    word = convertRationalNumberListToFloat(word)
    X_df = pd.DataFrame()
    for i in range(4):
        x = []
        for j in range(i, len(word), 4):
            x.append(word[j])
        if i == 0:
            X_df['open'] = x
        elif i == 1:
            X_df['high'] = x
        elif i == 2:
            X_df['low'] = x
        else:
            X_df['close'] = x

    X_df['series_id'] = [0 for _ in range(X_df.shape[0])]
    X_df['measurement_no'] = [i for i in range(X_df.shape[0])]
    y_df = pd.DataFrame()
    y_df['series_id'] = [0]
    y_df['label'] = [0]
    #faking the y in MyDataSet, since don't know the answer
    X_DataSet = MyDataSet(X_df, y_df, X_df.shape[0])
    X = DataLoader(X_DataSet, batch_size=1, shuffle=False, drop_last=True)
    return X

def askRNN(word:list) -> bool:
    testloader = convertRationalNumToRNNCompatible(word)
    rnnReply = test_model(rnn_model, testloader)
    if len(rnnReply) > 1:
        raise Exception(f"check the Query...since class prediction size is: {len(rnnReply)}")
    return rnnReply[0]

def membershipQuery(word: list, printing=True) -> bool:
    # expects a list of RationalNumbers
    if len(word) and type(word[0]) != type(RationalNumber(None, None)):
        raise Exception("membershipQuery was called with the list: " + str(word) + "\n of type: " + str(type(word)))

    if len(word) % 4 != 0 or len(word)==0:
        rnnReply = False
    else:
        rnnReply = askRNN(word)

    if rnnReply:
        if printing:
            print(f"membershipQuery: {word} is in the language.")
    else:
        if printing:
            print(f"membershipQuery: {word} is not in the language.")
    return rnnReply

def get_index_for_cex_if_any(y_batch, rnnReply):
    matched_count = 0
    first_unmatch_pos = -1
    for i, (actual, pred) in enumerate(zip(y_batch, rnnReply)):
        if int(actual) == int(pred):
            matched_count += 0
        else:
            if first_unmatch_pos == -1:
                first_unmatch_pos = i
            # print(first_unmatch_pos)
    percentageCorrect = (matched_count/len(y_batch)) * 100
    print(f"Accuracy during Equiv. check is {percentageCorrect}")
    return first_unmatch_pos

def statisticalEquivalenceQuery(automaton: RationalNominalAutomata) -> tuple:
    global samples
    # numberOfExamples = 100

    print("Checking equivalence of the following automaton:")
    print(automaton)
    seq_length = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timeSeriesData = TimeSeriesDataSet(
                        # file_list=["../data/BATS_GOOG_tVprocessed.csv"],
                          file_list=["../data/BATS_AMZN_tVprocessed.csv"],
                                       target_col='label', seq_length=seq_length, split=False)
    testloader, _ = timeSeriesData.get_loaders(batch_size=32)
    rnnReply = test_model(rnn_model, testloader)

    for i, batch_data in enumerate(testloader):
        X_batch = batch_data['data'].to(device)
        print(f"before flattening: {X_batch.size()}")
        X_batch = X_batch.flatten()
        print(f"After flattening: {X_batch.size()}")
        y_batch = batch_data['label'].to(device).long()
        y_batch = y_batch.float().unsqueeze(1)
        y_batch = y_batch.flatten()
        word = convertFloatToRationalNumberList(X_batch.numpy().tolist())
        hypothesisIsMember = automaton.accepts(word)
        # isMember = rnnReply[i]
        isMember = y_batch[i]
        if isMember != hypothesisIsMember:
            # print(f"The languages are not equivalent, a counterexample is: {word}")
            print(f"The languages are not equivalent, found a counterexample")
            if isMember:
                print("The word was rejected, but is in the language.")
            else:
                print("The word was accepted, but is not in the language.")
            return word
        else:
            if isMember:
                # print(str(word) + " was correctly accepted")
                print(f"The word was correctly accepted")
            else:
                # print(str(word) + " was correctly rejected")
                print(f"The word was correctly rejected")

    # cex_index = get_index_for_cex_if_any(y_batch, rnnReply)

    # if cex_index != -1:
    #     return testloader[cex_index].batch_data # tensor:


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
    #     # numberOfExamples -= len(p)
    # print(f"The languages appear to be  equivalent after checking {len(p)} random examples")
    return None

def main() -> None:
    learnedAutomaton = learn(membershipQuery, statisticalEquivalenceQuery)
    print(learnedAutomaton)
    gTComparison.statistics()


if __name__ == "__main__":
    main()
