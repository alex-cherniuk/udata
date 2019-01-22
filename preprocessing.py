import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class DataFrameWrapper:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def _columns_to_list(self, columns=None):
        if columns is None:
            return self.dataframe.columns.tolist()
        elif type(columns) in [str, int, float]:
            return [columns]
    
    def replace(self, old, new, columns=None, inplace=False):
        columns = self._columns_to_list(columns)
        new_dataframe = self.dataframe.copy(deep=True)
        for row_index, row in new_dataframe.iterrows():
            for column in columns:
                if pd.isna(old) and pd.isna(row[column]):
                    new_dataframe[column].iloc[row_index] = new
                elif row[column] == old:
                    new_dataframe[column].iloc[row_index] = new
        if inplace:
            self.dataframe = new_dataframe
        else:
            return new_dataframe
        
    def replace_by(self, old, by, columns=None, inplace=False):
        columns = self._columns_to_list(columns)        
        new_dataframe = self.dataframe.copy(deep=True)
        
        if by in ['mean', 'median']:
            new = eval(f'self.dataframe.{by}()')
        elif by == 'mode':
            new = eval(f'self.dataframe.{by}().iloc[0]')
        else:
            return
        for column in columns:
            result = self.replace(old, new[column], columns=column, inplace=False)
            if result is not None:
                new_dataframe[column] = result[column]
        if inplace:
            self.dataframe = new_dataframe
        else:
            return new_dataframe
        
    def drop_na(self, columns=None, axis=0, inplace=False):
        columns = self._columns_to_list(columns)        
        new_dataframe = self.dataframe.copy(deep=True)
        
        for row_index, row in new_dataframe.iterrows():
            for column in columns:
                if pd.isna(row[column]):
                    if axis == 0:
                        new_dataframe.drop([row_index], axis=0, inplace=True)
                        break
                    elif axis == 1:
                        new_dataframe.drop([column], axis=1, inplace=True)
                        columns.remove(column)
                    else:
                        raise Exception(f'Wrong axis! (axis = {axis})')
        if inplace:
            self.dataframe = new_dataframe
        else:
            return new_dataframe
        
    def linear_replace(self, inplace=False, columns=None):
        if not inplace:
            new_dataframe = self.dataframe.copy(deep=True)
        numerical_data = self.dataframe.select_dtypes(include=['float', 'int'])
        columns = numerical_data.columns.tolist()
        linear_regression = LinearRegression()

        for column in columns:
            X_train = numerical_data.dropna(axis=0)
            y_train = X_train[column]
            X_train.drop(column, axis=1, inplace=True)            
            X_pred = numerical_data[numerical_data[column].isnull()].drop(column, axis=1)
            if X_pred.empty:
                continue
            linear_regression.fit(X_train, y_train)
            y_pred = iter(linear_regression.predict(X_pred))
            for row_index, row in new_dataframe.iterrows():
                if pd.isna(value[column]):
                    if not inplace:
                        new_dataframe[column].iloc[row_index] = next(y_pred)
                    else:
                        self.dataframe[column].iloc[row_index] = next(y_pred)
        if not inplace:
            return new_dataframe
        
    def normalize(self, inplace=False):
        if not inplace:
            new_dataframe = self.dataframe.copy(deep=True)
        numerical_data = self.dataframe.select_dtypes(include=['float', 'int'])
        columns = numerical_data.columns.tolist()

        for column in columns:
            numerical_data[column] = numerical_data[column] - numerical_data[column].mean()
            numerical_data[column] /= np.std(numerical_data[column])
            if not inplace:
                new_dataframe[column] = numerical_data[column]
        if inplace:
            self.dataframe = new_dataframe
        else:
            return new_dataframe

    def scale(self, inplace=False):
        if not inplace:
            new_dataframe = self.dataframe.copy(deep=True)
        numerical_data = self.dataframe.select_dtypes(include=['float', 'int'])
        columns = numerical_data.columns.tolist()

        for column in columns:
            minimum = numerical_data[column].min()
            maximum = numerical_data[column].max()
            numerical_data[column] = (numerical_data[column] - minimum) / (maximum - minimum)
            if not inplace:
                new_dataframe[column] = numerical_data[column]
        if inplace:
            self.dataframe = new_dataframe
        else:
            return new_dataframe

    def distance(self, columns, pred_data=None, source=None):
        dataframe = self.dataframe[columns].copy(deep=True) if source is not None else source
        distance_list = []
        for row_index, row in dataframe.iterrows():
            distance = 0
            for column in columns:
                if pd.notna(pred_data[column]):
                    distance += np.square(pred_data[column] - row[column])
            distance_list.append(np.sqrt(distance))
        dataframe['Distance'] = distance_list
        return dataframe.sort_values(["Distance"], kind='heapsort')

    def knn_replace(self, inplace=False, columns=None, k=3):
        if not inplace:
            new_dataframe = self.dataframe.copy(deep=True)
        numerical_data = self.dataframe.select_dtypes(include=['float', 'int'])
        columns = numerical_data.columns.tolist()
        for column in columns:
            X_pred = numerical_data[numerical_data[column].isnull()].copy(deep=True)
            X_pred['target'] = 0
            X_train = numerical_data[column].notna()
            for row_index, row in X_pred.iterrows():
                distance = self.distance(columns, row, X_train)[1:k + 1]
                X_pred.target.loc[row_index] = distance[column].mean()
            y_pred = iter(X_pred.target)
            for row_index, row in self.dataframe.iterrows():
                if pd.isna(row[column]):
                    if not inplace:
                        new_dataframe[column].iloc[row_index] = next(y_pred)
                    else:
                        self.dataframe[column].iloc[row_index] = next(y_pred)
        if not inplace:
            return new_dataframe