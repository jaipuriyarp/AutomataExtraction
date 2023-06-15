class GTComparison:

    def __init__(self, memQ_for_GT):
        self.queries = {}
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
        if currKey in self.queries.keys():
            actual_ans =  self.queries[currKey]
        else:
            actual_ans = self.memQ_for_GT(word)
            self.queries[currKey] = (actual_ans)

        # queries counting for statistics
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
                    self.queries_incorrectlyAnswered.add(currKey)

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
                    self.queries_incorrectlyAnswered.add(currKey)
                else:
                    self.true_negative += 1
                    self.true_negative_EquivQ += 1

        return actual_ans

    def statistics(self):
        '''This function shows statistics'''

        print('*' * 40)
        print(f"Total Number of queries asked                                      : {len(self.queries)}")
        print(f"Total Number of membership queries asked                           : {self.num_pos_memQ + self.num_neg_memQ}")
        print(f"Total Number of positive membership queries asked                  : {self.num_pos_memQ}")
        print(f"Total Number of negative membership queries asked                  : {self.num_neg_memQ}")

        print(f"Total Number of queries asked during Equivalence check             : {self.num_pos_EquivQ + self.num_neg_EquivQ}")
        print(f"Total Number of positive queries asked during Equivalence check    : {self.num_pos_EquivQ}")
        print(f"Total Number of negative queries asked during Equivalence check    : {self.num_neg_EquivQ}")

        print('*' * 40)
        print(f"Total Number of queries answered by RNN correctly                  : {self.true_positive + self.true_negative}")
        print(f"Accuracy %                                                         : "
              f"{(self.true_positive + self.true_negative)/ (self.num_pos_memQ + self.num_neg_memQ + self.num_pos_EquivQ + self.num_neg_EquivQ)}")

        ## membership queries:
        print('*' * 40)
        print(f"Total Number of membership queries answered by RNN correctly          : "
              f"{(self.true_positive_memQ + self.true_negative_memQ)}")
        print(f"Accuracy %                                                            : "
              f"{(self.true_positive_memQ + self.true_negative_memQ) / (self.num_pos_memQ + self.num_neg_memQ)}")
        print(f"Total Number of positive membership queries answered by RNN correctly : {self.true_positive_memQ}")
        print(f"Accuracy                                                              : {self.true_positive_memQ/self.num_pos_memQ}")
        print(f"Total Number of negative membership queries answered by RNN correctly : {self.true_negative_memQ}")
        print(f"Accuracy                                                              : {self.true_negative_memQ / self.num_neg_memQ}")

        ## Equivalence queries:
        print('*' * 40)
        print(f"Total Number of queries answered during equivalence check by RNN correctly          : "
            f"{(self.true_positive_EquivQ + self.true_negative_EquivQ)}")
        print(f"Accuracy %                                                                          : "
              f"{(self.true_positive_EquivQ + self.true_negative_EquivQ) / (self.num_pos_EquivQ + self.num_neg_EquivQ)}")
        print(f"Total Number of positive queries answered during equivalence check by RNN correctly : {self.true_positive_EquivQ}")
        print(f"Accuracy                                                                            : "
              f"{self.true_positive_EquivQ/ self.num_pos_EquivQ}")
        print(f"Total Number of negative queries answered during equivalence check by RNN correctly : {self.true_negative_EquivQ}")
        print(f"Accuracy                                                                            : "
              f"{self.true_negative_EquivQ / self.num_neg_EquivQ}")



