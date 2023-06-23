import sys

pathsToInclude = ["../../TheoryOfEquality/vLStarForRationalAutomata/"]
for path in pathsToInclude:
    sys.path.append(path)

from vLStar import RationalNumber

def checkType(word: list):
    if len(word) and not all([type(i) == type(RationalNumber(None,None)) for i in word]):
        raise Exception("membershipQuery was called with the list: "+str(word)+"\n of type: "+str(type(word)))

def Lang_is_aStar(word: list):
    ''' This functions gives correct/ ground Truth for the language a* i.e. (a)^*'''

    checkType(word)

    if len(set(word)) > 1: #language rejecting criteria
        return False

    return True

def Lang_is_abSeq(word: list):
    ''' This functions gives correct/ ground Truth for the language abSeq i.e. (ab)^n where n>0'''

    checkType(word)

    if len(word) % 2 == 0 and not (False in [word[i] == word[i+2] for i in range(len(word)-2)]) \
    and (len(word)>1 and word[0] != word[1]): #language acceptance criteria
        return True

    return False

def Lang_is_abSeq_OddaEvenb(word:list):
    ''' This functions gives correct/ ground Truth for the language a^nb^m where
    where n is an odd number and b is a an even number where b must be greater then 0 '''

    checkType(word)

    s  = list(set(word))
    if len(s) != 2:
        return False

    if s[0] != word[0]:
        s[0], s[1] = s[1], s[0]

    i, count_a = 0, 0
    while i < len(word) and s[0] == word[i]:
        count_a += 1
        i += 1
    if all([word[i]==s[1] for i in range(i, len(word))]) and count_a % 2 == 1 \
            and word.count(s[1]) %2 == 0: #language acceptance criteria
        return True

    return False

def Lang_is_noTrigrams(word: list):
    '''This functions gives correct/ ground Truth for the language which accepts any
    string, w which doesn't contain any trigrams.'''

    checkType(word)

    if len(word) < 3:
        return True

    for i in range(len(word) - 2):
        if (word[i] == word[i+1]) and (word[i+1] == word[i+2]): # language rejection criteria
            return False

    return True

def Lang_is_abBothEven(word: list):
    '''This functions gives correct/ ground Truth for the language which accepts any
        string, w which contains only two unique words and is even in number'''

    checkType(word)

    if len(set(word)) > 2:
        return False

    uniq_word = list(set(word))
    if (word.count(uniq_word[0]) % 2 == 0) and  (word.count(uniq_word[0]) % 2 == 0):
        return True

    return False

def Lang_is_aMod3b(word: list):
    '''This functions gives correct/ ground Truth for the language which accepts any
            string, w when #_a(w) equiv_3 to #_b(w)'''

    checkType(word)
    raise Exception("NOT IMPLEMENTED YET")

def Lang_is_aStarbStaraStarbStar(word: list):
    '''This functions gives correct/ ground Truth for the language which accepts language
    a*b*a*b*'''

    checkType(word)
    raise Exception("NOT IMPLEMENTED YET")



