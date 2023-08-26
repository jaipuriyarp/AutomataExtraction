class GTComparison:

    def __init__(self, memQ_for_GT=None):
        self.queries = {}
        self.qCount  = 0
        #actual count:
        self.num_pos_memQ = 0
        self.num_neg_memQ = 0
        self.num_pos_EquivQ = 0
        self.num_neg_EquivQ = 0

        #RNN:
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0 # model predicted +ve but was actually -ve
        self.false_negative = 0 # model predicted -ve but was actually +ve

        #RNN mem Queries Only:
        self.true_positive_memQ = 0
        self.true_negative_memQ = 0
        self.false_positive_memQ = 0  # model predicted +ve but was actually -ve
        self.false_negative_memQ = 0  # model predicted -ve but was actually +ve

        # RNN mem Queries Only:
        self.true_positive_EquivQ = 0
        self.true_negative_EquivQ = 0
        self.false_positive_EquivQ = 0  # model predicted +ve but was actually -ve
        self.false_negative_EquivQ = 0  # model predicted -ve but was actually +ve

        self.queries_incorrectlyAnswered = set()
        self.memQ_for_GT = memQ_for_GT #function for finding ground truth for language


    def getGT(self, word:list , RNN_ans, memQ): # word: list[RationalNUmbers]
        '''This function is used to compare the GT with RNN answer on the fly with the function provided to it'''
        currKey = tuple(word)
        self.qCount += 1
        if currKey in self.queries.keys():
            actual_ans =  self.queries[currKey]
        else:
            if self.memQ_for_GT is not None:
                actual_ans = self.memQ_for_GT(word)
                self.queries[currKey] = (actual_ans)
            else:
                actual_ans = None

        # queries counting for statistics
        if actual_ans is not None:
            if actual_ans: # case: positive case
                if memQ: # case: membership Queries
                    self.num_pos_memQ += 1
                    if RNN_ans:
                        self.true_positive += 1
                        self.true_positive_memQ += 1
                    else:
                        self.false_negative += 1
                        self.false_negative_memQ += 1
                        self.queries_incorrectlyAnswered.add(currKey)

                else: # case: Equivalence check
                    self.num_pos_EquivQ += 1
                    if RNN_ans:
                        self.true_positive += 1
                        self.true_positive_EquivQ += 1
                    else:
                        self.false_negative += 1
                        self.false_negative_EquivQ += 1
                        # self.queries_incorrectlyAnswered.add(currKey)

            else: # case: negative
                if memQ: # case: membership Queries
                    self.num_neg_memQ  += 1
                    if RNN_ans:
                        self.false_positive += 1
                        self.false_positive_memQ += 1
                        self.queries_incorrectlyAnswered.add(currKey)
                    else:
                        self.true_negative += 1
                        self.true_negative_memQ += 1

                else: # case: Equivalence check
                    self.num_neg_EquivQ += 1
                    if RNN_ans:
                        self.false_positive += 1
                        self.false_positive_EquivQ +=1
                        # self.queries_incorrectlyAnswered.add(currKey)
                    else:
                        self.true_negative += 1
                        self.true_negative_EquivQ += 1
        else:
            if memQ:  # case: membership Queries
                self.num_pos_memQ += 1
            else:
                self.num_pos_EquivQ += 1

        return actual_ans

    def statistics(self):
        '''This function shows statistics'''
        if self.memQ_for_GT is None:
            print(f"WARNING: There is no ground truth values, so every mem and Eqiuv queries "
                  "are shown as positive memQ and Equiv Queries.")

        print('*' * 100)
        print(f"Total Number of queries asked                                      : {self.qCount}")
        print(f"Total number of pos queries asked                                  : {self.num_pos_memQ + self.num_pos_EquivQ}")
        print(f"Total number of neg queries asked                                  : {self.num_neg_memQ + self.num_neg_EquivQ}")
        print(f"Total number of queries answered correctly by RNN                  : {self.true_positive + self.true_negative}")
        print(f"Accuracy                                                           : {(self.true_positive + self.true_negative) /self.qCount}")
        print('*' * 100)

        print(f"Total Number of membership queries asked                           : {self.num_pos_memQ + self.num_neg_memQ}")
        print(f"Total Number of positive membership queries asked                  : {self.num_pos_memQ}")
        print(f"Total Number of negative membership queries asked                  : {self.num_neg_memQ}")

        print(f"Total Number of queries asked during Equivalence check             : {self.num_pos_EquivQ + self.num_neg_EquivQ}")
        print(f"Total Number of positive queries asked during Equivalence check    : {self.num_pos_EquivQ}")
        print(f"Total Number of negative queries asked during Equivalence check    : {self.num_neg_EquivQ}")


        ## membership queries:
        print('*' * 100)
        print(f"Total Number of membership queries answered by RNN correctly          : "
              f"{(self.true_positive_memQ + self.true_negative_memQ)}")
        print(f"Accuracy of membership queries answered                               : "
              f"{(self.true_positive_memQ + self.true_negative_memQ) / (self.num_pos_memQ + self.num_neg_memQ)}")
        print(f"Total Number of positive membership queries answered by RNN correctly : {self.true_positive_memQ}")
        print(f"Accuracy of +ve membership queries                                    : {self.true_positive_memQ/self.num_pos_memQ}")
        print(f"Total Number of negative membership queries answered by RNN correctly : {self.true_negative_memQ}")
        print(f"Accuracy of -ve membership queries                                    : {self.true_negative_memQ / self.num_neg_memQ}")

        ## Equivalence queries:
        print('*' * 100)
        print(f"Total Number of queries answered during equivalence check by RNN correctly          : "
            f"{(self.true_positive_EquivQ + self.true_negative_EquivQ)}")
        if self.num_pos_EquivQ > 0 or self.num_neg_EquivQ > 0 :
            print(f"Accuracy of equivalence queries answered                                            : "
                f"{(self.true_positive_EquivQ + self.true_negative_EquivQ) / (self.num_pos_EquivQ + self.num_neg_EquivQ)}")
        print(f"Total Number of positive queries answered during equivalence check by RNN correctly : {self.true_positive_EquivQ}")
        if self.num_pos_EquivQ > 0:
            print(f"Accuracy of +ve queries asked during equivalence check                              : "
                 f"{self.true_positive_EquivQ/ self.num_pos_EquivQ}")
        print(f"Total Number of negative queries answered during equivalence check by RNN correctly : {self.true_negative_EquivQ}")
        if self.num_neg_EquivQ > 0:
            print(f"Accuracy of -ve queries asked during equivalence check                              : "
                 f"{self.true_negative_EquivQ / self.num_neg_EquivQ}")

        print(f"Queries incorrectly answered by RNN is:{self.queries_incorrectlyAnswered}")



