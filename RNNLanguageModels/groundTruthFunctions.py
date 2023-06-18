import sys

pathsToInclude = ["../../TheoryOfEquality/vLStarForRationalAutomata/"]
for path in pathsToInclude:
    sys.path.append(path)

from vLStar import RationalNumber

def Lang_is_aStar(word: list):
    ''' This functions gives correct/ ground Truth for the language a* i.e. (a)^*'''
    if len(word) and type(word[0]) != type(RationalNumber(None,None)):
        raise Exception("membershipQuery was called with the list: "+str(word)+"\n of type: "+str(type(word)))

    if len(set(word)) > 1: #language rejecting criteria
        return False

    return True

def Lang_is_abSeq(word: list):
    ''' This functions gives correct/ ground Truth for the language abSeq i.e. (ab)^n where n>0'''
    if len(word) and type(word[0]) != type(RationalNumber(None,None)):
        raise Exception("membershipQuery was called with the list: "+str(word)+"\n of type: "+str(type(word)))

    if len(word) % 2 == 0 and not (False in [word[i] == word[i+2] for i in range(len(word)-2)]) \
    and (len(word)>1 and word[0] != word[1]): #language acceptance criteria
        return True

    return False


