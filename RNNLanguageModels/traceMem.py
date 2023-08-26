# importing the module
import tracemalloc

from transformerParenthesesTraining import start

# code or function for which memory
# has to be monitored
# def app():
#     lt = []
#     for i in range(0, 100000):
#         lt.append(i)


# starting the monitoring
tracemalloc.start()

# function call
start()

# displaying the memory
print(tracemalloc.get_traced_memory())

# stopping the library
tracemalloc.stop()