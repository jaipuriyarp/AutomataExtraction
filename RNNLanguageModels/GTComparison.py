import pandas as pd
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

        self.queries_incorrectlyAnswered = []
        self.memQ_for_GT = memQ_for_GT #function for finding ground truth for language
        self.query_elapsed_time_dict = {}
        self.round_digits_upto = 4

    def queriesCount(self):
        return self.qCount

    def addAdversarialExample(self, word:list):
        if word not in self.queries_incorrectlyAnswered:
            self.queries_incorrectlyAnswered.append(word)
    def save_elapsed_time_for_query(self, word:list, time:float):
        if tuple(word) not in self.query_elapsed_time_dict.keys():
            self.query_elapsed_time_dict[tuple(word)] = time

    def display_adversarial_query_time_relation(self, file_name=None):
        adversarial_example_col, adversarial_example_time_col, adversarial_example_length_col = [], [], []
        for key in self.query_elapsed_time_dict.keys():
            adversarial_example_col.append(key)
            adversarial_example_time_col.append(self.query_elapsed_time_dict[key])
            adversarial_example_length_col.append(len(key))
        data = { "Adversarial Examples" : adversarial_example_col,
                 "Time(s)" : adversarial_example_time_col,
                 "Length of Adversarial Examples" : adversarial_example_length_col
        }
        df = pd.DataFrame(data)
        print(df)
        if file_name is not None:
            df.to_csv(file_name, index=False)


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
                        self.addAdversarialExample(word)

                else: # case: Equivalence check
                    self.num_pos_EquivQ += 1
                    if RNN_ans:
                        self.true_positive += 1
                        self.true_positive_EquivQ += 1
                    else:
                        self.false_negative += 1
                        self.false_negative_EquivQ += 1
                        self.addAdversarialExample(word)

            else: # case: negative
                if memQ: # case: membership Queries
                    self.num_neg_memQ  += 1
                    if RNN_ans:
                        self.false_positive += 1
                        self.false_positive_memQ += 1
                        self.addAdversarialExample(word)
                    else:
                        self.true_negative += 1
                        self.true_negative_memQ += 1

                else: # case: Equivalence check
                    self.num_neg_EquivQ += 1
                    if RNN_ans:
                        self.false_positive += 1
                        self.false_positive_EquivQ +=1
                        self.addAdversarialExample(word)
                    else:
                        self.true_negative += 1
                        self.true_negative_EquivQ += 1
        else:
            if memQ:  # case: membership Queries
                self.num_pos_memQ += 1
            else:
                self.num_pos_EquivQ += 1

        return actual_ans

    def statistics(self, file_name=None):
        '''This function shows statistics'''
        if self.memQ_for_GT is None:
            print(f"WARNING: There is no ground truth values, so every mem and Eqiuv queries "
                  "are shown as positive memQ and Equiv Queries.")
        total_accuracy = 0
        pos_memQ_accuray, neg_memQ_accuracy, total_memQ_accuracy = 0, 0, 0
        pos_EquivQ_accuray, neg_EquivQ_accuracy, total_EquivQ_accuracy = 0, 0, 0
        header_column = ["No. of queries asked", "No. of queries answered correctly", "No. of queries answered incorrectly",
                         "Accuracy"]
        if self.num_pos_memQ:
            pos_memQ_accuray = round(self.true_positive_memQ/self.num_pos_memQ, ndigits=self.round_digits_upto)

        if self.num_neg_memQ:
            neg_memQ_accuracy = round(self.true_negative_memQ/self.num_neg_memQ, ndigits=self.round_digits_upto)

        if self.num_pos_memQ or self.num_neg_memQ:
            total_memQ_accuracy = round((self.true_positive_memQ + self.true_negative_memQ) / (self.num_pos_memQ + self.num_neg_memQ),
                                        ndigits=self.round_digits_upto)

        mem_pos_column = [self.num_pos_memQ, self.true_positive_memQ,
                          self.num_pos_memQ - self.true_positive_memQ, pos_memQ_accuray]
        mem_neg_column = [self.num_neg_memQ, self.true_negative_memQ,
                          self.num_neg_memQ - self.true_negative_memQ, neg_memQ_accuracy]
        mem_total_column = [self.num_pos_memQ + self.num_neg_memQ, self.true_positive_memQ + self.true_negative_memQ,
                            (self.num_pos_memQ + self.num_neg_memQ) - (self.true_positive_memQ + self.true_negative_memQ),
                            total_memQ_accuracy]
        #Equiv:
        if self.num_pos_memQ:
            pos_EquivQ_accuray = round(self.true_positive_EquivQ / self.num_pos_EquivQ, ndigits=self.round_digits_upto)

        if self.num_neg_memQ:
            neg_EquivQ_accuracy = round(self.true_negative_EquivQ / self.num_neg_EquivQ, ndigits=self.round_digits_upto)

        if self.num_pos_EquivQ or self.num_neg_EquivQ:
            total_EquivQ_accuracy = round((self.true_positive_EquivQ + self.true_negative_EquivQ) / (self.num_pos_EquivQ + self.num_neg_EquivQ),
                                          ndigits=self.round_digits_upto)

        equiv_pos_column = [self.num_pos_EquivQ, self.true_positive_EquivQ,
                          self.num_pos_EquivQ - self.true_positive_EquivQ, pos_EquivQ_accuray]
        equiv_neg_column = [self.num_neg_EquivQ, self.true_negative_EquivQ,
                          self.num_neg_EquivQ - self.true_negative_EquivQ, neg_EquivQ_accuracy]
        equiv_total_column = [self.num_pos_EquivQ + self.num_neg_EquivQ,
                            self.true_positive_EquivQ + self.true_negative_EquivQ,
                            (self.num_pos_EquivQ + self.num_neg_EquivQ) - (
                                        self.true_positive_EquivQ + self.true_negative_EquivQ),
                            total_EquivQ_accuracy]

        if (self.num_pos_memQ + self.num_neg_memQ + self.num_pos_EquivQ + self.num_neg_EquivQ):
            total_accuracy = (self.true_positive_memQ + self.true_negative_memQ + self.true_positive_EquivQ + self.true_negative_EquivQ) / \
                             (self.num_pos_memQ + self.num_neg_memQ + self.num_pos_EquivQ + self.num_neg_EquivQ)

        total_pos_count = [self.num_pos_memQ + self.num_neg_memQ + self.num_pos_EquivQ + self.num_neg_EquivQ,
                           self.true_positive_memQ + self.true_negative_memQ + self.true_positive_EquivQ + self.true_negative_EquivQ,
                           (self.num_pos_memQ + self.num_neg_memQ + self.num_pos_EquivQ + self.num_neg_EquivQ) -
                           (self.true_positive_memQ + self.true_negative_memQ + self.true_positive_EquivQ + self.true_negative_EquivQ),
                           total_accuracy]

        data = {
            "Details"                : header_column,
            "+ve Membership Query"   : mem_pos_column,
            "-ve Membership Query"   : mem_neg_column,
            "Total Membership Query" : mem_total_column,
            "+ve Equivalence Query"  : equiv_pos_column,
            "-ve Equivalence Query"  : equiv_neg_column,
            "Total Equivalence Query": equiv_total_column,
            "Total count"            : total_pos_count
        }

        df = pd.DataFrame(data)
        print(f"INFO: Statistics from GTComparison:")
        print(df)
        if file_name is not None:
            df.to_csv(file_name, index=False)




