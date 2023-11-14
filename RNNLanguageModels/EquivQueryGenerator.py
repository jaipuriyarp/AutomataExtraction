import random

# upper_limit_of_sequence_generation = 1000
def queryGenerator(upper_limit_of_sequence_generation=1000, max_length = 20, current_query_count=None,
                   langOnQ=True):
    # decreasing one count for emptyWord, to check only once
    per_length_count = (upper_limit_of_sequence_generation - 1) / max_length
    if current_query_count:
        per_length_count = (upper_limit_of_sequence_generation - current_query_count) / max_length

    print(f"INFO: per_length_queries to generate : {per_length_count} "
          f"rounded of to: {round(per_length_count)} "
          f"previous count is: {current_query_count}")

    per_length_count = round(per_length_count)

    start, end = 1, (2 ^ 63 - 1)
    for l in range(max_length + 1):
        num_examples = per_length_count
        if langOnQ:
            #For Tomita:
            if l == 0:
                num_examples = 1
            while num_examples > 0:
                yield [random.randint(start, end) for _ in range(l)]
                num_examples -= 1
        else:
            #For balanced Parentheses:
            # choice_list = ['(', ')', 'x']
            choice_list = ['(', ')']
            while num_examples > 0:
                yield (''.join([random.choice(choice_list) for _ in range(k)]))
                num_examples -= 1


def generateOneQuery(max_length, no_of_times_called=None, lang=1):
    if no_of_times_called % 1000 == 0:
        print(f"INFO : No. of times..query Generated: {no_of_times_called}")
    start, end = 1, (2 ^ 63 - 1)
    if no_of_times_called < 1000:
        max_length = 5
    elif no_of_times_called < 5000:
        max_length = 10
    else:
        max_length = max_length
    if lang == 8:
        choice_list = ['(', ')']
        yield (''.join([random.choice(choice_list) for _ in range(0, max_length)]))
    else:
        yield [random.randint(start, end) for _ in range(random.randint(0, max_length))]


# Usage:
# for q in queryGenerator():
#     print(q)



