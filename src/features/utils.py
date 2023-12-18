import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats as st
import category_encoders as ce
#----------------------CategoricalTransformer----------------------

class CategoricalTransformer():
    def __init__(self, max_unique_values = 20):
        self.max_unique_values = max_unique_values
        self.__selected_features = None

    def fit(self, X):
        # Identify features with 20 or fewer unique values
        self.__selected_features = X.columns[X.nunique() <= self.max_unique_values]
        return self

    def transform(self, X):
        X_transformed = X.copy()
        # Convert selected features to categorical
        X_transformed[self.__selected_features] = X_transformed[self.__selected_features].astype('object')
        return X_transformed
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



#----------------------MissingValuesFiller----------------------

class MissingValuesFiller():
    def __init__(self, numeric_strategy='mean', categorical_strategy='most_frequent'):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.__numeric_cols = None
        self.__categorical_cols = None
        self.__mean_numeric_series = None
        self.__mode_categorical_series = None

    def fit(self, X):
        # Separate numeric and categorical columns
        self.__numeric_cols = X.select_dtypes(include='number').columns
        self.__categorical_cols = X.select_dtypes(include='object').columns
        self.__mean_numeric_series= X[self.__numeric_cols].agg(self.numeric_strategy)
        self.__mode_categorical_series= X[self.__categorical_cols].mode().iloc[0] if self.categorical_strategy  == 'most_frequent' else self.categorical_strategy

    def transform(self, X):
        X_transformed = X.copy()
         # Fill missing values in numeric columns
        X_transformed[self.__numeric_cols] = X[self.__numeric_cols].fillna(self.__mean_numeric_series)

        # Fill missing values in categorical columns
        X_transformed[self.__categorical_cols] = X[self.__categorical_cols].fillna(self.__mode_categorical_series)

        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



#----------------------DropHighCorrelationFeatures----------------------


class DropHighCorrelationFeatures():
    def __init__(self, threshold = 0.85):
        self.threshold = threshold
        self.to_drop_cols = None

    def fit(self, X):
        # Calculate the correlation matrix
        correlation_matrix = X.corr().abs()

        # Create a mask to identify highly correlated features
        upper_triangle_mask = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

        # Identify features with correlation above the threshold
        to_drop = [column for column in upper_triangle_mask.columns if any(upper_triangle_mask[column] > self.threshold)]

        self.to_drop_cols = to_drop



    def transform(self, X):
        X_transformed = X.copy()

        # Drop highly correlated features
        X_transformed = X_transformed.drop(columns=self.to_drop_cols)

        return X_transformed
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

#----------------------SplitterByType----------------------

class SplitterByType():
    def __init__(self):
        self.__numeric_cols = None
        self.__categorical_cols = None

    def fit(self, X):
        # Identify numeric and categorical columns
        self.__numeric_cols = X.select_dtypes(include='number').columns
        self.__categorical_cols = X.select_dtypes(include='object').columns



    def transform(self, X):
        X_transformed = X.copy()

        # Create DataFrames based on column types
        numeric_df = X[self.__numeric_cols].copy()
        categorical_df = X[self.__categorical_cols].copy()

        return numeric_df, categorical_df

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


#----------------------ConstantsAndHighCardinalityDropper----------------------
class ConstantsAndHighCardinalityDropper():
    def __init__(self,max_unique_values=2000):
        self.max_unique_values = max_unique_values
        self.__non_constant_cols = None
        self.__high_cardinality_cols = None

    def fit(self, X):
        # Identify numeric and categorical columns
        self.__non_constant_cols = X.columns[X.nunique() > 1]
        self.__high_cardinality_cols = X.columns[X.nunique() > self.max_unique_values]


    def transform(self, X):
        X_transformed = X.copy()

        # Get non constants columns
        X_transformed = X_transformed[self.__non_constant_cols]

        #drop high cardinality columns
        X_transformed = X_transformed.drop(columns=self.__high_cardinality_cols)

        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


#----------------------ConstantsAndHighCardinalityDropper----------------------
class FrequencyEncoder():
    def __init__(self):
        self.categorical_features = None
        #dictionary with values of the categorical features as keys and dictionary of frequencies
        self.mapper = {}

    def fit(self, X):
        # Identify numeric and categorical columns
        self.categorical_features = X.select_dtypes('object').columns

        for cat_col in self.categorical_features:
            class_values_frequency = {}
            for class_value in X[cat_col].unique():
                class_values_frequency[class_value] = sum(X[cat_col]==class_value)/X.shape[0]
            self.mapper[cat_col] = class_values_frequency


    def transform(self, X):
        X_transformed = X.copy()

        # apply frequencies mapping
        for cat_col in self.categorical_features:
            X_transformed[cat_col] = X_transformed[cat_col].map(self.mapper[cat_col])

        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)