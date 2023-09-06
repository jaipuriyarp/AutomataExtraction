import random

# upper_limit_of_sequence_generation = 1000
max_length = 20
def queryGenerator(upper_limit_of_sequence_generation=1000, max_length = 20, current_query_count=None):
    # decreasing one count for emptyWord, to check only once
    per_length_count = (upper_limit_of_sequence_generation - 1) / max_length
    if current_query_count:
        per_length_count = (upper_limit_of_sequence_generation - current_query_count) / max_length

    print(f"INFO: per_length_queries to generate : {per_length_count} "
          f"rounded of to: {int(per_length_count)}"
          f"previous count is: {current_query_count}")

    per_length_count = int(per_length_count)


    start, end = 1, (2 ^ 63 - 1)
    for l in range(max_length + 1):
        num_examples = per_length_count
        if l == 0:
            num_examples = 1
        while num_examples > 0:
            yield [random.randint(start, end) for _ in range(l)]
            num_examples -= 1

# Usage:
# for q in queryGenerator():
#     print(q)


