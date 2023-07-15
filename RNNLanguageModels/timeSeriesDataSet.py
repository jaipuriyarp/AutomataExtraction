import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

verbose = 1
def debug(verbose_level:int, str:str):
    if verbose >= verbose_level:
        print(str)

class MyDataSet(Dataset):
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, seq_len:int):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        # self.drop_column_names = ['series_id', 'measurement_no']
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
    def __init__(self, file_list:list, target_col:str, seq_length:int, split=True):
        # self.data = data
        self.file_list = file_list
        self.target_col = target_col
        self.seq_length = seq_length
        self.split = split
        # self.numerical_cols = list(set(data.columns) - set(target_col))
        # self.numerical_cols = None
        self.preprocessor = StandardScaler()
        self.train_num = None
        self.test_num = None
        self.train_X = pd.DataFrame()
        self.train_y = pd.DataFrame()
        self.test_X = pd.DataFrame()
        self.test_y = pd.DataFrame()

    def load_data(self, file:str):
        data = pd.read_csv(file)
        return data
    def preprocess_data(self, data:pd.DataFrame):
        '''This function preprocess using Standard Scaler i.e. it fits the data to be have mean 0 and
        splits the data into train and test data
        :return: returns X_train, X_test, y_train, y_test of datatype as pandas.DataFrame'''
        X = data.drop(self.target_col, axis=1)
        y = data[self.target_col]

        debug(3, f"X is {X}")
        debug(3, f"y is {y}")
        # debug(0, f"shape of X is {X.shape()}")
        # debug(0, f"shape of y is {X.shape()}")
        # self.preprocess = ColumnTransformer(
        #     [("scaler", StandardScaler(), self.numerical_cols)],
        #      # ("encoder", OneHotEncoder(), self.categorical_cols)],
        #     remainder="passthrough")
        X_train = X
        y_train = y
        if self.split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)


        # X_train = self.preprocessor.fit_transform(X_train)
        # X_test = self.preprocessor.transform(X_test)
        # debug(0, f"INFO: Training and testing data set has been transformed to be around mean.")

        debug(3, f"DEBUG: After fit, X_train is {X_train}")
        debug(3, f"DEBUG: After fit, y_train is {y_train.values}")

        debug(2, f"DEBUG: After fit, shape of X_train is {X_train.shape}")
        debug(2, f"DEBUG: After fit, shape of y_train is {y_train.shape}")

        if self.split:
            debug(3, f"DEBUG: After fit, X_test is {X_test}")
            debug(3, f"DEBUG: After fit, y_test is {y_test.values}")

            debug(2, f"DEBUG: After fit, shape of X_test is {X_test.shape}")
            debug(2, f"DEBUG: After fit, shape of y_test is {y_test.shape}")

        if self.split:
            return X_train, y_train, X_test, y_test
        else:
            return X, y, None, None

    def convert_to_multipleTimeSeries(self, X:pd.DataFrame, y:pd.DataFrame):
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

        if dfXNew['measurement_no'].max() != self.seq_length-1:
            raise Exception (f"ERROR: measurement_no is never equal to {self.seq_length}")

        debug(0,f"INFO: Number of series created for this time series data is: {series_idCount-1}"
                f" and measurement_no for each series is: {dfXNew['measurement_no'].max()}")
        return dfXNew, dfYNew
        # return MyDataSet(dfXNew, dfYNew, self.seq_length)
    def accumulate_multipleTimeSeries(self, X:pd.DataFrame, y:pd.DataFrame, train:bool):
        if train:
            self.train_X = pd.concat([self.train_X, X], ignore_index=True)
            self.train_y = pd.concat([self.train_y, y], ignore_index=True)
            debug(0, f"INFO: Accumulation: shape of train_X is {self.train_X.shape}")
            debug(0, f"INFO: Accumulation: shape of train_y is {self.train_y.shape}")
        else:
            self.test_X = pd.concat([self.test_X, X], ignore_index=True)
            self.test_y = pd.concat([self.test_y, y], ignore_index=True)
            debug(0, f"INFO: Accumulation: shape of test_X is {self.test_X.shape}")
            debug(0, f"INFO: Accumulation: shape of test_y is {self.test_y.shape}")

    def get_MyDatSet(self, train:bool):
        if train:
            return MyDataSet(self.train_X, self.train_y, self.seq_length)
        else:
            return MyDataSet(self.test_X, self.test_y, self.seq_length)

    def get_loaders(self, batch_size: int, write=False):
        '''
        Preprocess and frame the dataset
        :param batch_size: batch size
        :return: DataLoaders associated to training and testing data
        '''
        for i,file in enumerate(self.file_list):
            debug(0, f"INFO: Started processing {i}st/nd/th file: {file}")
            data = self.load_data(file)
            X_train, y_train, X_test, y_test = self.preprocess_data(data)

            train_X_df, train_y_df = self.convert_to_multipleTimeSeries(X_train, y_train)
            self.accumulate_multipleTimeSeries(train_X_df, train_y_df, train=True)
            if write:
                debug(0, "DEBUG: writing..")
                train_X_df.to_csv(Path("../data/", str(i)+"_train_X.csv"), index=False)
                train_y_df.to_csv(Path("../data/", str(i)+"_train_y.csv"), index=False)

            train_dataset = self.get_MyDatSet(train=True)
            self.train_num = len(train_dataset)
            debug(0, f"INFO: size of train_dataset is : {len(train_dataset)} "
                     f"and type of train_dataset is : {type(train_dataset)}")

            train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


            if (X_test is not None) and (y_test is not None):
                test_X_df, test_y_df = self.convert_to_multipleTimeSeries(X_test, y_test)
                self.accumulate_multipleTimeSeries(test_X_df, test_y_df, train=False)
                test_dataset = self.get_MyDatSet(train=False)
                self.test_num = len(test_dataset)
                debug(0, f"INFO: size of test_dataset is : {len(test_dataset)} "
                         f"and type of test_dataset is : {type(test_dataset)}")
                test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            else:
                test_iter = None

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
