# Language accepted is: balanced parentheses.
# ():
# ( : 10 : 2
# ) : 01 : 1
import sys
import os
import random

modelDir = "../"
pathsToInclude = ["../../vLStarForRationalAutomata/", modelDir, "../RNNLanguageModels"]
for path in pathsToInclude:
    sys.path.append(path)

from vLStar import RationalNumber, RationalNominalAutomata, learn

from rnnInterface import RNNInterface
from GTComparison import GTComparison
from groundTruthFunctions import is_balanced_parenthesis
from balancedParanthesis import generate_balanced_parentheses, generate_unbalanced_parentheses\
        ,generate_unbalanced_parantheses_from_posL, \
    generate_unbalanced_parentheses_up_to_depth, generate_one_parantheses_up_to_depth,  encode_sequence

RNNModelName = "modelRNN_lang8_balancedParenthesis.pt"
RNNModelPath = os.path.join(modelDir, "models", RNNModelName)
maxDepth = 12


rnnInterface = RNNInterface(rnn_model_path=RNNModelPath, input_size=2)
gTComparison = GTComparison(is_balanced_parenthesis)

def membershipQuery(word: list, printing=True) -> bool:
    # expects a list of RationalNumbers
    if len(word) and type(word[0]) != type(RationalNumber(0,1)) :
        raise Exception("membershipQuery was called with the list: " + str(word) + "\n of type: " + str(type(word)))

    rnnReply = rnnInterface.askRNN(word, paranthesesLang=True)
    word = rnnInterface.getRNNCompatibleInputFromRationalNumber(word, paranthesesLang=True)
    # print(f"The query is: {word}")
    Qreply = gTComparison.getGT(word, rnnReply, printing)
    # print (Qreply)
    if rnnReply:
        if printing:
            print(f"membershipQuery: {word} is in the language.")
    else:
        if printing:
            print(f"membershipQuery: {word} is not in the language.")
    if printing:
        return rnnReply
    return Qreply

def getQueriesForEquivalenceQuery(k:int):
    if k == 0:
        k = random.randint(1, maxDepth)
    p = generate_balanced_parentheses(k)
    p += generate_one_parantheses_up_to_depth(k)
    p += generate_unbalanced_parentheses_up_to_depth(k)
    p += generate_unbalanced_parantheses_from_posL(p)
    # return [[RationalNumber(0,1) if char=="(" else RationalNumber(1,1) for char in word] for word in p]
    return p
def statisticalEquivalenceQuery(automaton: RationalNominalAutomata) -> tuple:
    global samples
    # numberOfExamples = 100

    print("Checking equivalence of the following automaton:")
    print(automaton)

    # for l in range(numberOfExamples):  # + numberOfEquivalenceQueriesAsked):
    k = 9
    # p = []
    # for i in range(k):
    #     p += getQueriesForEquivalenceQuery(i)
    # p += getQueriesForEquivalenceQuery(0)
    for i in range(k):
        chkQ = [generate_balanced_parentheses, generate_unbalanced_parentheses, generate_one_parantheses_up_to_depth]
        for j in range(len(chkQ)):
            for stringQ in chkQ[j](i):
            # for word in p:
                word = [RationalNumber(1, 1) if char == '(' else RationalNumber(0, 1) for char in stringQ]
                isMember = membershipQuery(word, False)
                hypothesisIsMember = automaton.accepts(word)

                if isMember != hypothesisIsMember:
                    print("The languages are not equivalent, a counterexample is: " + str(word))
                    if isMember:
                        print("The word was rejected, but is in the language.")
                    else:
                        print("The word was accepted, but is not in the language.")
                    return word
                else:
                    if isMember:
                        print(str(word) + " was correctly accepted")
                    else:
                        print(str(word) + " was correctly rejected")


        # numberOfExamples -= len(p)
    print(f"The languages appear to be  equivalent after checking upto depth {k} random examples")
    return None

def main() -> None:
    learnedAutomaton = learn(membershipQuery, statisticalEquivalenceQuery, paranthesesLang=True)
    print(learnedAutomaton)
    gTComparison.statistics()


if __name__ == "__main__":
    main()
