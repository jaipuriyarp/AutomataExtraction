# Language accepted is: (ab)^n, where n is a Natural number excluding 0.
import sys
import os
import random
import torch

modelDir = "../"
pathsToInclude = ["../../TheoryOfEquality/vLStarForRationalAutomata/", modelDir]
for path in pathsToInclude:
    sys.path.append(path)

from vLStar import RationalNumber, RationalNominalAutomata, learn
from wordPattern import load_model, createRNNModel, load_RNN_model, encode_sequence, test_RNN

RNNModelName = "modelRNN.pt"
RNNModelPath =  os.path.join(modelDir, "models", RNNModelName)
word2vecModelPath = os.path.join(modelDir, "models", "GoogleNews-vectors-negative300.bin")

UpperBound = 3000000 #number of words in Google's word2vec
debug = True


samples = [[RationalNumber(0,1)],
           [RationalNumber(0,1),RationalNumber(0,1)],
           [RationalNumber(0,1),RationalNumber(1,1)],
           [RationalNumber(1,1),RationalNumber(0,1)],
           [RationalNumber(1,1),RationalNumber(1,1)],
           [RationalNumber(0,1),RationalNumber(0,1),RationalNumber(0,1)],
           [RationalNumber(0,1),RationalNumber(0,1),RationalNumber(1,1)],
           [RationalNumber(0,1),RationalNumber(1,1),RationalNumber(0,1)],
           [RationalNumber(0,1),RationalNumber(1,1),RationalNumber(1,1)],
           [RationalNumber(1,1),RationalNumber(0,1),RationalNumber(0,1)],
           [RationalNumber(1,1),RationalNumber(0,1),RationalNumber(1,1)],
           [RationalNumber(1,1),RationalNumber(1,1),RationalNumber(0,1)],
           [RationalNumber(1,1),RationalNumber(1,1),RationalNumber(1,1)],
           [RationalNumber(0,1),RationalNumber(0,1),RationalNumber(0,1),RationalNumber(0,1)],
           [RationalNumber(0,1),RationalNumber(0,1),RationalNumber(0,1),RationalNumber(0,1),RationalNumber(0,1)],
           [RationalNumber(0,1),RationalNumber(0,1),RationalNumber(0,1),RationalNumber(0,1),RationalNumber(0,1),RationalNumber(0,1)]]

class RNNInterface:

    def __init__(self):
        self.w2v_model = load_model(word2vecModelPath, limit=3000)
        # Set the device for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.RNN_model = createRNNModel(self.device)
        # Load the model saved already in the models folder
        self.RNN_model , _ = load_RNN_model(self.RNN_model, RNNModelPath)

    def convertRationalNumberListToWordList(self, string: list):
        is_denominator_same = all(x.getDenominator() == string[0].getDenominator() for x in string)
        if is_denominator_same or string == []:
            if len(string) > 1 and \
               all((x == string[1] and string[1] != string[0]) for x,i in zip(string,range(len(string))) if i%2 == 1):
                wordL = [self.w2v_model.index_to_key[x.getNumerator()] if i%2==0 else '#' for x,i in zip(string,range(len(string)))]
            else:
                wordL = [self.w2v_model.index_to_key[x.getNumerator()] for x in string]
            return wordL
        else:
            raise("ERROR: NOT IMPLEMENTED YET!") #TODO: implement this later

    def askRNN(self, NumList: list):
        if debug: ##Hack for right now to match '#'
            wordL = self.convertRationalNumberListToWordList(NumList)
            print(f"The converted query to RNN is:{wordL}")
            X = [encode_sequence(wordL, self.w2v_model)]

        else:
            X = [self.self.w2v_model[x.getNumerator()] for x in NumList]
        RNN_reply =  test_RNN(self.RNN_model, X, None, self.device, False)
        if len(RNN_reply) != 1:
            raise Exception(f"Since, the student asks teacher one query at a time, "
                            "the length of answer from Teacher i.e. RNN should be exactly One")
        return RNN_reply[0]


rnnInterface = RNNInterface()
def membershipQuery(word : list, printing = True) -> bool:
    #expects a list of RationalNumbers
    if len(word) and type(word[0]) != type(RationalNumber(None,None)):
        raise Exception("membershipQuery was called with the list: "+str(word)+"\n of type: "+str(type(word)))
    print(f"The query is: {word}")
    Qreply = rnnInterface.askRNN(word)
    print (Qreply)
    if Qreply:
        if printing:
            print(f"membershipQuery: {word} matches the pattern abab.")
    else:
        if printing:
            print(f"membershipQuery: {word} is not in the language.")
    return Qreply

def statisticalEquivalenceQuery(automaton : RationalNominalAutomata) -> tuple:
    global samples
    numberOfExamples = 100

    print("Checking equivalence of the following automaton:")
    print(automaton)

    for word in samples:
        isMember = membershipQuery(word,False)
        hypothesisIsMember = automaton.accepts(word)

        if isMember != hypothesisIsMember:
            print("The languages are not equivalent, a counterexample is: " +str(word))
            if isMember:
                print("The word was rejected, but is in the language.")
            else:
                print("The word was accepted, but is not in the language.")
            return (False,word)
        else:
            if isMember:
                print(str(word) + " was correctly accepted")
            else:
                print(str(word) + " was correctly rejected")
    
    for l in range(numberOfExamples): # + numberOfEquivalenceQueriesAsked):
        length = 1 + int(l/20)
        numerators   = [random.randint(1,5) for i in range(length)] #random.sample(range(0, 5), length)
        denominators = [1] * length  #random.sample(range(1, 10 + 2 * l), length)
        word = [RationalNumber(numerators[i],denominators[i]) for i in range(length)]
        
        isMember = membershipQuery(word,False)
        hypothesisIsMember = automaton.accepts(word)

        if isMember != hypothesisIsMember:
            print("The languages are not equivalent, a counterexample is: " +str(word))
            if isMember:
                print("The word was rejected, but is in the language.")
            else:
                print("The word was accepted, but is not in the language.")
            return (False,word)
        else:
            if isMember:
                print(str(word) + " was correctly accepted")
            else:
                print(str(word) + " was correctly rejected")
    print("The languages appear to be  equivalent after checking "+str(numberOfExamples)+" random examples")
    return (True,None)

def main() -> None :
    learnedAutomaton = learn(membershipQuery, statisticalEquivalenceQuery)
    print(learnedAutomaton)
	
if __name__ == "__main__":
    main()
