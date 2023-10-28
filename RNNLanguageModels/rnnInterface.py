import sys
modelDir = "../"
pathsToInclude = ["../../TheoryOfEquality/vLStarForRationalAutomata/", modelDir, "."]
for path in pathsToInclude:
    sys.path.append(path)

# from languageGeneratorOnQ import encode_sequence
from balancedParanthesis import encode_sequence as encode_parantheses
from languageGeneratorOnQ_from_file import encode_sequence
from vLStar import RationalNumber
from rnnModel import RNNModel

class RNNInterface:

    def __init__(self, rnn_model_path: str, input_size: int, hidden_size=64, output_size=1, num_layers=2,
                 maxlength=20):
        # self.w2v_model = load_model(word2vec_model_path, limit=None)
        RNN_model = RNNModel(input_size=input_size, hidden_size=hidden_size,
                             output_size=output_size, num_layers=num_layers, model_name=rnn_model_path.split("/")[-1])
        # Load the model saved already in the models folder
        RNN_model.load_RNN_model(rnn_model_path)
        print ("RNN loaded successfully! {RNN_model}")
        self.rnn_model = RNN_model
        self.maxlength = maxlength

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

    # def convertRationalNumberListToWordList(self, string: list):
    #     #TODO: Write a new function to convert directly Rational Num to Tensor, rather than converting to word first. in RNN
    #     is_denominator_same = all(x.getDenominator() == string[0].getDenominator() for x in string)
    #     if not is_denominator_same:
    #         #making the denom same for all rational Numbers by calaculating lcm
    #         denominator_list = (x.getDenominator() for x in string)
    #         lcm = find_lcm(denominator_list[0], denominator_list[1])
    #         for i in range(2, len(denominator_list)):
    #             lcm = find_lcm(lcm, denominator_list[i])
    #         string = makeEqRationalNumber(string, lcm)
    #
    #     wordL = [self.w2v_model.index_to_key[x.getNumerator()] for x in string]
    #     return wordL

    def convertRationalNumberListToFloat(self, rationalNumList: list) -> list:
        '''This function converts Rational number to floating point number'''
        if len(rationalNumList) > 0 and type(rationalNumList[0]) is float:
            return rationalNumList
        return [num.getNumerator()/num.getDenominator() for num in rationalNumList]


    def convertRationalNumberListToParanthesesStr(self, rationalNumList: list) -> list:
        '''This function converts Rational number to a string of parentheses'''
        '''TODO: catch: what if first parentheses is closing instead of opening parentheses'''
        if type(rationalNumList) == str:
            return rationalNumList
        else:
            # if rationalNumList is str and all([x=="(" or x==")" for x in rationalNumList]): # already converted??
            #     return rationalNumList
            # print(f"rationalNumList: {rationalNumList}")
            x = list(set(rationalNumList))

            paranthesesEncode = {}
            if len(x) > 1:
                # if x[0] > x[1]:
                #     x[0], x[1] = x[1], x[0]
                x.sort()
                paranthesesEncode = {x[0] : ")", x[1] : "("}
            elif len(x) == 1:
                paranthesesEncode = {x[0]: ")"}


            stringQuery = ""
            for num in rationalNumList:
                if num in paranthesesEncode.keys():
                    stringQuery += paranthesesEncode[num]
                else:
                    stringQuery += "x"

            return stringQuery
            # return [paranthesesEncode for x in rationalNumList]

    def getRNNCompatibleInputFromRationalNumber(self, numList:list, paranthesesLang:bool): #-> tensor
        if paranthesesLang:
            return self.convertRationalNumberListToParanthesesStr(numList)
        return self.convertRationalNumberListToFloat(numList)

    def askRNN(self, numList: list, paranthesesLang=False) -> bool:
        wordL = self.getRNNCompatibleInputFromRationalNumber(numList, paranthesesLang)
        # print(wordL)
        if paranthesesLang:
            if len(set(numList)) > 2 or 'x' in wordL:
                return False
            X = [encode_parantheses(wordL, self.maxlength)]
        else:
            X = [encode_sequence(wordL, self.maxlength)]

        # print(f"The converted query to RNN is:{wordL}")

        RNN_reply =  self.rnn_model.test_RNN(X, None)
        if len(RNN_reply) != 1:
            raise Exception(f"Since, the student asks teacher one query at a time, "
                            "the length of answer from Teacher i.e. RNN should be exactly One")
        return RNN_reply[0]