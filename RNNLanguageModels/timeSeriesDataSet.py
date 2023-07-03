import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

verbose = 2
def debug(verbose_level, str):
    if verbose >= verbose_level:
        print(str)

class MyDataSet(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        print(f"INFO: In MyDataSet: Shape of X is: {X.shape}")
        print(f"INFO: In MyDataSet: Shape of y is: {y.shape}")
        print(f"INFO: In MyDataSet: Number of time series: {len(self.y)}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        data = self.X.iloc[self.seq_len * idx: self.seq_len * (idx + 1)].drop(
            columns=['series_id', 'measurement_no']).values.astype('float32')
        label = self.y.iloc[idx]['label']


        return_dict = {'data': data, 'label': label}
        return return_dict

class TimeSeriesDataSet():
    def __init__(self, data, target_col, seq_length):
        self.data = data
        self.target_col = target_col
        self.seq_length = seq_length
        self.numerical_cols = list(set(data.columns) - set(target_col))
        self.preprocessor = StandardScaler()
        self.train_num = None
        self.test_num = None

    def preprocess_data(self):
        '''This function preprocess using Standard Scaler i.e. it fits the data to be have mean 0 and
        splits the data into train and test data
        :return: returns X_train, X_test, y_train, y_test of datatype as pandas.DataFrame'''
        X = self.data.drop(self.target_col, axis=1)
        y = self.data[self.target_col]

        debug(3, f"X is {X}")
        debug(3, f"y is {y}")
        # debug(0, f"shape of X is {X.shape()}")
        # debug(0, f"shape of y is {X.shape()}")
        # self.preprocess = ColumnTransformer(
        #     [("scaler", StandardScaler(), self.numerical_cols)],
        #      # ("encoder", OneHotEncoder(), self.categorical_cols)],
        #     remainder="passthrough")

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

        # X_train = self.preprocessor.fit_transform(X_train)
        # X_test = self.preprocessor.transform(X_test)

        # debug(0, f"Training and testing data set has been transformed to be around mean.")

        debug(3, f"After fit, X_train is {X_train}")
        debug(3, f"After fit, y_train is {y_train.values}")
        debug(3, f"After fit, X_test is {X_test}")
        debug(3, f"After fit, y_test is {y_test.values}")

        debug(2, f"INFO: After fit, shape of X_train is {X_train.shape}")
        debug(2, f"INFO: After fit, shape of y_train is {y_train.shape}")
        debug(2, f"INFO: After fit, shape of X_test is {X_test.shape}")
        debug(2, f"INFO: After fit, shape of y_test is {y_test.shape}")
        # debug(1, f"X_train type: {type(X_train)}, y_train type: {type(y_train)}")

        return X_train, X_test, y_train, y_test

    def convert_to_multipleTimeSeries(self, X, y):
        '''
        This function prepares the X and y to multiple time series.
        :param X: X data after preprocessing
        :param y: y data after preprocessing
        :return: MyDataSet object
        '''

        # converting numpy.ndarray to panda DataFrame
        X = pd.DataFrame(X)
        # y = pd.DataFrame(y)

        step_size = 2
        series_idCount = 0
        dfXNew = pd.DataFrame()
        dfYNew = pd.DataFrame()
        label = []

        for i in range(0, X.shape[0], step_size):
            r = X[i:i + self.seq_length]
            debug(3, f"r.shape: {r.shape}")
            if r.shape[0] != self.seq_length:
                shortageBy = self.seq_length - r.shape[0]
                r = X[i - shortageBy:i + self.seq_length]
            series_id = []
            measurement_no = []
            X_new = pd.DataFrame()
            for j in range(0, r.shape[0]):
                series_id.append(series_idCount)
                measurement_no.append(j)
            X_new['series_id'] = series_id
            X_new['measurement_no'] = measurement_no
            for col in list(X.columns):
                X_new[col] = [x for x in r[col]]
            debug(3, f"X_new is \n {X_new}")
            dfXNew = pd.concat([dfXNew, X_new], ignore_index=True)

            label.append(y[i:i + self.seq_length].iloc[-1])
            series_idCount += 1

        dfYNew['series_id'] = [i for i in range(0, len(label))]
        dfYNew['label'] = label

        return MyDataSet(dfXNew, dfYNew, self.seq_length)

    def get_loaders(self, batch_size: int):
        '''
        Preprocess and frame the dataset
        :param batch_size: batch size
        :return: DataLoaders associated to training and testing data
        '''
        X_train, X_test, y_train, y_test = self.preprocess_data()

        train_dataset = self.convert_to_multipleTimeSeries(X_train, y_train)
        test_dataset = self.convert_to_multipleTimeSeries(X_test, y_test)

        self.train_num = len(train_dataset)
        self.test_num = len(test_dataset)


        debug(0, f"size of train_dataset is : {len(train_dataset)} "
                 f"and type of train_dataset is : {type(train_dataset)}")
        debug(0, f"size of test_dataset is : {len(test_dataset)} "
                 f"and type of test_dataset is : {type(test_dataset)}")


        train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        return train_iter, test_iter

    def get_train_length(self):
        if self.train_num is None:
            print(f"WARNING: The training length has not been assigned yet, please run "
                  f"get_loader function before this function")
        return self.train_num

    def get_test_length(self):
        if self.test_num is None:
            print(f"WARNING: The training length has not been assigned yet, please run "
                  f"get_loader function before this function")
        return self.test_num
