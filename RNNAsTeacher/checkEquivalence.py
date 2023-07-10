import sys
import os
import random
from itertools import product
sys.path.append(os.path.abspath("../../"))

from vLStar import RationalNumber, RationalNominalAutomata
maxInteger = (2^64) - 1

class CheckEquivalence:
    def __init__(self, depth=2, num_of_RationalNumber=2, automaton=None, membershipQuery=None):
        self.depth = depth
        self.num_of_RationalNumber =  num_of_RationalNumber
        self.automaton = automaton
        self.membershipQuery = membershipQuery
        self.numQ = 0

    def setAutomaton(self, automaton: RationalNominalAutomata):
        self.automaton = automaton

    def setMembershipQuery(self, membershipQuery):
        self.membershipQuery = membershipQuery

    def getQueriesCount(self):
        return self.numQ

    def compareMemQAndAutomata(self, word: list):
        isMember, hypothesisIsMember = None, None
        if self.membershipQuery is not None:
            isMember = self.membershipQuery(word, False)
        if self.automaton is not None:
            hypothesisIsMember = self.automaton.accepts(word)
        # print(f"MemQ Answer: {isMember} and hypothesis answer: {hypothesisIsMember}")

        if isMember != hypothesisIsMember:
            print(f"The languages are not equivalent, a counterexample is: {str(word)}")
            if isMember:
                print(f"The word was rejected, but is in the language.")
            else:
                print(f"The word was accepted, but is not in the language.")
            return word
        else:
            if isMember:
                print(f"{str(word)} was correctly accepted")
            else:
                print(f"{str(word)} was correctly rejected")

        return None

    def generatePermutations(self, k: int, num: list): # returns iterator
        for length in range(1, k + 1):
            for p in product(num, repeat=length): #returns iterator
                yield list(p)

    def generateQueries(self):
        numerators = random.choices(range(0, maxInteger), k=self.num_of_RationalNumber)
        numerators.sort()
        num = [RationalNumber(i, 1) for i in numerators]
        # check all permutations of num upto fixedlength:
        x = [query for i in range(self.depth + 1) for query in self.generatePermutations(i, num)]
        self.numQ = len(x)
        return x

    def askQueries(self) -> bool:
        '''checkEquivalence'''
        for query in self.generateQueries():
            cex = self.compareMemQAndAutomata(query)
            if cex is not None:
                return cex
        return None