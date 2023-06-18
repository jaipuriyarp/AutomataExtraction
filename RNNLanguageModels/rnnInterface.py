import sys
modelDir = "../"
pathsToInclude = ["../../TheoryOfEquality/vLStarForRationalAutomata/", modelDir, "."]
for path in pathsToInclude:
    sys.path.append(path)

from languageGenerator import load_model, encode_sequence
from vLStar import RationalNumber
from rnnModel import RNNModel

class RNNInterface:

    def __init__(self, word2vec_model_path, rnn_model_path):
        self.w2v_model = load_model(word2vec_model_path, limit=None)
        RNN_model = RNNModel(input_size=300, hidden_size=64,
                             output_size=1, num_layers=2, model_name=rnn_model_path.split("/")[-1])
        # Load the model saved already in the models folder
        RNN_model.load_RNN_model(rnn_model_path)
        print ("RNN loaded successfully! {RNN_model}")
        self.rnn_model = RNN_model

    def find_lcm(self, num1: int, num2: int):
        max = max(num1, num2)
        min = min(num1, num2)
        rem = max % min
        while (rem != 0):
            max = min
            min = rem
            rem = max % min
        gcd = min
        lcm = (num1*num2) / int(gcd)
        return int(lcm)

    def makeEqRationalNumber(self, string, lcm):
        return [RationalNumber(x.getNumerator()*(lcm/x.getDenominator()), lcm) for x in string]

    def convertRationalNumberListToWordList(self, string: list):
        #TODO: Write a new function to convert directly Rational Num to Tensor, rather than converting to word first. in RNN
        is_denominator_same = all(x.getDenominator() == string[0].getDenominator() for x in string)
        if not is_denominator_same:
            #making the denom same for all rational Numbers by calaculating lcm
            denominator_list = (x.getDenominator() for x in string)
            lcm = find_lcm(denominator_list[0], denominator_list[1])
            for i in range(2, len(denominator_list)):
                lcm = find_lcm(lcm, denominator_list[i])
            string = makeEqRationalNumber(string, lcm)

        wordL = [self.w2v_model.index_to_key[x.getNumerator()] for x in string]
        return wordL

    def askRNN(self, NumList: list):
        wordL = self.convertRationalNumberListToWordList(NumList)
        print(f"The converted query to RNN is:{wordL}")
        X = [encode_sequence(wordL, self.w2v_model)]
        RNN_reply =  self.rnn_model.test_RNN(X, None)
        if len(RNN_reply) != 1:
            raise Exception(f"Since, the student asks teacher one query at a time, "
                            "the length of answer from Teacher i.e. RNN should be exactly One")
        return RNN_reply[0]