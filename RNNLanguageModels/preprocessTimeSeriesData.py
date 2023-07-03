import argparse
import pandas as pd
import math
from pathlib import Path

data_dir = '../data/'
def load_data(file_name:str):
    data = pd.read_csv(file_name)
    return data
def add_label(data:pd.DataFrame , colName:str):
    col = data[colName]
    label = [1 if math.isnan(x) else 0 for x in col]
    data['label'] = label
    return data

def clean_data(data:pd.DataFrame):
    column_names = list(data.columns.values)
    columnToPreserve = ['open', 'high', 'low', 'close', 'label']
    for x in column_names:
        if x not in columnToPreserve:
            data = data.drop(columns=x)
    return data

def preprocess_data(data: pd.DataFrame, colNameForLabel='Gann Swing High Plots-Triangles Down Top of Screen'):
    dataLabelled = add_label(data, colNameForLabel)
    cleanData = clean_data(dataLabelled)
    return cleanData


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str)
    args = parser.parse_args()
    df = load_data(args.file_name)

    final_data = preprocess_data(df)
    newFile = args.file_name.strip(".csv").strip("../orig_data") + "processed.csv"
    final_data.to_csv(Path(data_dir, newFile), index=False)
    print("INFO: File: " + newFile + "saved to the dir: " + data_dir)



