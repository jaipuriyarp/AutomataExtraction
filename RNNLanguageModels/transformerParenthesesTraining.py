import random

##Transformer for learning balanced parantheses using SimpleTransformers

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import random
import tracemalloc
from balancedParanthesis import generateExamples
from groundTruthFunctions import is_balanced_parenthesis

maxlength=24
verbose=0
num_epochs = 100

def debug(verbose_level, str):
    if verbose >= verbose_level:
        print(str)

def prepare_data(maxlength):
    posL, negL = generateExamples(maxlength=maxlength)
    debug(2, f"posL: {posL}, negL: {negL}")
    train_data = [[x, 1] for x in posL]
    train_data += [[x, 0] for x in negL]
    random.shuffle(train_data)
    debug(1, f"train_data: {train_data}")
    train_df = pd.DataFrame(train_data)
    train_df.columns = ["text", "labels"]
    print(f"INFO: Size of training data: {train_df.size}")
    return train_df

def get_eval_data(maxlength, numexamples):
    t = []
    for _ in range(int(numexamples)):
        s = ""
        for _ in range(random.randint(0, maxlength)):
            s+= random.choice(["(", ")"])
        t.append(s)
    posL, negL = generateExamples(maxlength-3)
    eval_data = [[x, int(is_balanced_parenthesis(x))] for x in t]
    eval_data += [[x, 1] for x in posL]
    eval_data += [[x, 0] for x in negL]
    debug(1, f"eval_data: {eval_data}")
    eval_df = pd.DataFrame(eval_data)
    eval_df.columns = ["text", "labels"]
    print(f"INFO: Size of test data: {eval_df.size}")
    return eval_df

def create_model_and_train(train_df, eval_df):
    # Optional model configuration
    model_args = ClassificationArgs(num_train_epochs=num_epochs)

    # Create a ClassificationModel
    model = ClassificationModel(
        "roberta", "roberta-base", args=model_args,
        use_cuda=False
    )

    # Train the model
    tracemalloc.start()
    model.train_model(train_df, output_dir="../model_transformer/",
                      eval_df=eval_df)
    print(tracemalloc.get_traced_memory())
    tracemalloc.start()

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print(result, model_outputs, wrong_predictions)

    # # Make predictions with the model
    # predictions, raw_outputs = model.predict(["Sam was a Wizard"])




# # Preparing train data
# train_data = [
#     ["Aragorn was the heir of Isildur", 1],
#     ["Frodo was the heir of Isildur", 0],
# ]
# train_df = pd.DataFrame(train_data)
# train_df.columns = ["text", "labels"]
#
# # Preparing eval data
# eval_data = [
#     ["Theoden was the king of Rohan", 1],
#     ["Merry was the king of Rohan", 0],
# ]
# eval_df = pd.DataFrame(eval_data)
# eval_df.columns = ["text", "labels"]
#
# # Optional model configuration
# model_args = ClassificationArgs(num_train_epochs=1)
#
# # Create a ClassificationModel
# model = ClassificationModel(
#     "roberta", "roberta-base", args=model_args
# )
#
# # Train the model
# model.train_model(train_df)
#
# # Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(eval_df)
#
# # Make predictions with the model
# predictions, raw_outputs = model.predict(["Sam was a Wizard"])

def start():
    train_df = prepare_data(maxlength)
    eval_df = get_eval_data(maxlength, 500)
    create_model_and_train(train_df, eval_df)

if __name__ == "__main__":
    start()



