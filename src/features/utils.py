import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats as st

#----------------------CategoricalTransformer----------------------

class CategoricalTransformer:
    """
    A transformer class for handling categorical features with a limited number of unique values.

    Parameters:
    - max_unique_values (int): Maximum number of unique values allowed (default is 20).

    Methods:
    - fit(X): Identify features with a number of unique values less than or equal to max_unique_values.
    - transform(X): Convert selected features to categorical type.
    - fit_transform(X): Combined fit and transform operations.

    Example:
    ```
    # Create an instance of CategoricalTransformer
    transformer = CategoricalTransformer(max_unique_values=15)

    # Fit and transform a DataFrame df
    df_transformed = transformer.fit_transform(df)
    ```

    Attributes:
    - max_unique_values (int): Maximum number of unique values allowed.
    - __selected_features (list): List of selected features based on fit operation.
    """
    def __init__(self, max_unique_values=20):
        self.max_unique_values = max_unique_values
        self.__selected_features = None

    def fit(self, X):
        """
        Identify features with a number of unique values less than or equal to max_unique_values.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - self: Returns an instance of the CategoricalTransformer class.
        """
        self.__selected_features = X.columns[X.nunique() <= self.max_unique_values]
        return self

    def transform(self, X):
        """
        Convert selected features to categorical type.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with selected features converted to categorical type.
        """
        X_transformed = X.copy()
        X_transformed[self.__selected_features] = X_transformed[self.__selected_features].astype('object')
        return X_transformed

    def fit_transform(self, X):
        """
        Combined fit and transform operations.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with selected features converted to categorical type.
        """
        self.fit(X)
        return self.transform(X)




#----------------------MissingValuesFiller----------------------

class MissingValuesFiller:
    """
    A transformer class for handling missing values in both numeric and categorical columns.

    Parameters:
    - numeric_strategy (str or callable): Strategy to fill missing values in numeric columns (default is 'mean').
    - categorical_strategy (str or callable): Strategy to fill missing values in categorical columns (default is 'most_frequent').

    Methods:
    - fit(X): Learn and store information needed for filling missing values in both numeric and categorical columns.
    - transform(X): Fill missing values in the input DataFrame using the learned strategies.
    - fit_transform(X): Combined fit and transform operations.

    Example:
    ```
    # Create an instance of MissingValuesFiller
    filler = MissingValuesFiller(numeric_strategy='median', categorical_strategy='constant')

    # Fit and transform a DataFrame df
    df_transformed = filler.fit_transform(df)
    ```

    Attributes:
    - numeric_strategy (str or callable): Strategy for filling missing values in numeric columns.
    - categorical_strategy (str or callable): Strategy for filling missing values in categorical columns.
    - __numeric_cols (Index): Numeric columns in the input DataFrame.
    - __categorical_cols (Index): Categorical columns in the input DataFrame.
    - __mean_numeric_series (Series): Mean values for filling missing values in numeric columns.
    - __mode_categorical_series (Series): Mode values for filling missing values in categorical columns.
    """
    def __init__(self, numeric_strategy='mean', categorical_strategy='most_frequent'):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.__numeric_cols = None
        self.__categorical_cols = None
        self.__mean_numeric_series = None
        self.__mode_categorical_series = None
        
    def fit(self, X):
        """
        Learn and store information needed for filling missing values in both numeric and categorical columns.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - self: Returns an instance of the MissingValuesFiller class.
        """
        # Separate numeric and categorical columns
        self.__numeric_cols = X.select_dtypes(include='number').columns
        self.__categorical_cols = X.select_dtypes(include='object').columns
        self.__mean_numeric_series = X[self.__numeric_cols].agg(self.numeric_strategy)
        self.__mode_categorical_series = X[self.__categorical_cols].mode().iloc[0] if self.categorical_strategy == 'most_frequent' else self.categorical_strategy 

    def transform(self, X):
        """
        Fill missing values in the input DataFrame using the learned strategies.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with missing values filled.
        """
        X_transformed = X.copy()
        # Fill missing values in numeric columns
        X_transformed[self.__numeric_cols] = X[self.__numeric_cols].fillna(self.__mean_numeric_series)
    
        # Fill missing values in categorical columns
        X_transformed[self.__categorical_cols] = X[self.__categorical_cols].fillna(self.__

        


#----------------------DropHighCorrelationFeatures----------------------


class DropHighCorrelationFeatures:
    """
    A transformer class for dropping highly correlated features based on a correlation threshold.

    Parameters:
    - threshold (float): Threshold for identifying highly correlated features (default is 0.85).

    Methods:
    - fit(X): Identify features with correlation above the specified threshold.
    - transform(X): Drop highly correlated features from the input DataFrame.
    - fit_transform(X): Combined fit and transform operations.

    Example:
    ```
    # Create an instance of DropHighCorrelationFeatures
    dropper = DropHighCorrelationFeatures(threshold=0.9)

    # Fit and transform a DataFrame df
    df_transformed = dropper.fit_transform(df)
    ```

    Attributes:
    - threshold (float): Threshold for identifying highly correlated features.
    - to_drop_cols (list): List of columns identified for dropping based on correlation threshold.
    """
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.to_drop_cols = None

    def fit(self, X):
        """
        Identify features with correlation above the specified threshold.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - self: Returns an instance of the DropHighCorrelationFeatures class.
        """
        # Calculate the correlation matrix
        correlation_matrix = X.corr().abs()
    
        # Create a mask to identify highly correlated features
        upper_triangle_mask = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
    
        # Identify features with correlation above the threshold
        to_drop = [column for column in upper_triangle_mask.columns if any(upper_triangle_mask[column] > self.threshold)]
        
        self.to_drop_cols = to_drop

    def transform(self, X):
        """
        Drop highly correlated features from the input DataFrame.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with highly correlated features dropped.
        """
        X_transformed = X.copy()
        
        # Drop highly correlated features
        X_transformed = X_transformed.drop(columns=self.to_drop_cols)
        
        return X_transformed

    def fit_transform(self, X):
        """
        Combined fit and transform operations.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with highly correlated features dropped.
        """
        self.fit(X)
        return self.transform(X)


#----------------------SplitterByType----------------------
class SplitterByType:
    """
    A transformer class for splitting a DataFrame into separate DataFrames based on column types.

    Methods:
    - fit(X): Identify numeric and categorical columns in the input DataFrame.
    - transform(X): Split the input DataFrame into separate DataFrames for numeric and categorical columns.
    - fit_transform(X): Combined fit and transform operations.

    Example:
    ```
    # Create an instance of SplitterByType
    splitter = SplitterByType()

    # Fit and transform a DataFrame df
    numeric_df, categorical_df = splitter.fit_transform(df)
    ```

    Attributes:
    - __numeric_cols (Index): Numeric columns in the input DataFrame.
    - __categorical_cols (Index): Categorical columns in the input DataFrame.
    """
    def __init__(self):
        self.__numeric_cols = None
        self.__categorical_cols = None

    def fit(self, X):
        """
        Identify numeric and categorical columns in the input DataFrame.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - self: Returns an instance of the SplitterByType class.
        """
        # Identify numeric and categorical columns
        self.__numeric_cols = X.select_dtypes(include='number').columns
        self.__categorical_cols = X.select_dtypes(include='object').columns

    def transform(self, X):
        """
        Split the input DataFrame into separate DataFrames for numeric and categorical columns.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame, pd.DataFrame: Separate DataFrames for numeric and categorical columns.
        """
        X_transformed = X.copy()
        
        # Create DataFrames based on column types
        numeric_df = X[self.__numeric_cols].copy()
        categorical_df = X[self.__categorical_cols].copy()
        
        return numeric_df, categorical_df

    def fit_transform(self, X):
        """
        Combined fit and transform operations.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame, pd.DataFrame: Separate DataFrames for numeric and categorical columns.
        """
        self.fit(X)
        return self.transform(X)




#----------------------ConstantsAndHighCardinalityDropper----------------------
class ConstantsAndHighCardinalityDropper:
    """
    A transformer class for dropping constant columns and columns with high cardinality.

    Attributes:
    - max_unique_values (int): Maximum number of unique values allowed for a non-constant column.
    - __non_constant_cols (Index): Columns with more than one unique value.
    - __high_cardinality_cols (Index): Columns with more unique values than the specified threshold.

    Methods:
    - fit(X): Identify non-constant columns and columns with high cardinality.
    - transform(X): Drop constant columns and columns with high cardinality from the input DataFrame.
    - fit_transform(X): Combined fit and transform operations.

    Example:
    ```
    # Create an instance of ConstantsAndHighCardinalityDropper
    dropper = ConstantsAndHighCardinalityDropper(max_unique_values=1500)

    # Fit and transform a DataFrame df
    df_transformed = dropper.fit_transform(df)
    ```

    """
    def __init__(self, max_unique_values=2000):
        self.max_unique_values = max_unique_values
        self.__non_constant_cols = None
        self.__high_cardinality_cols = None

    def fit(self, X):
        """
        Identify non-constant columns and columns with high cardinality.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - self: Returns an instance of the ConstantsAndHighCardinalityDropper class.
        """
        # Identify non-constant columns and columns with high cardinality
        self.__non_constant_cols = X.columns[X.nunique() > 1]
        self.__high_cardinality_cols = X.columns[X.nunique() > self.max_unique_values]

    def transform(self, X):
        """
        Drop constant columns and columns with high cardinality from the input DataFrame.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with constant columns and high cardinality columns dropped.
        """
        X_transformed = X.copy()
        
        # Get non-constant columns
        X_transformed = X_transformed[self.__non_constant_cols]

        # Drop high cardinality columns
        X_transformed = X_transformed.drop(columns=self.__high_cardinality_cols)
        
        return X_transformed

    def fit_transform(self, X):
        """
        Combined fit and transform operations.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with constant columns and high cardinality columns dropped.
        """
        self.fit(X)
        return self.transform(X)

#----------------------CategoryToFrequencyConverter----------------------
class FrequencyEncoder:
    """
    A transformer class for encoding categorical features based on their frequencies.

    Attributes:
    - categorical_features (Index): Categorical columns in the input DataFrame.
    - mapper (dict): Dictionary with values of the categorical features as keys and a dictionary of frequencies.

    Methods:
    - fit(X): Learn and store the frequencies of each unique value in the categorical features.
    - transform(X): Encode categorical features based on the learned frequencies.
    - fit_transform(X): Combined fit and transform operations.

    Example:
    ```
    # Create an instance of FrequencyEncoder
    encoder = FrequencyEncoder()

    # Fit and transform a DataFrame df
    df_transformed = encoder.fit_transform(df)
    ```

    """
    def __init__(self):
        self.categorical_features = None
        self.mapper = {}

    def fit(self, X):
        """
        Learn and store the frequencies of each unique value in the categorical features.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - self: Returns an instance of the FrequencyEncoder class.
        """
        # Identify numeric and categorical columns
        self.categorical_features = X.select_dtypes('object').columns
        
        for cat_col in self.categorical_features:
            class_values_frequency = {}
            for class_value in X[cat_col].unique():
                class_values_frequency[class_value] = sum(X[cat_col] == class_value) / X.shape[0]
            self.mapper[cat_col] = class_values_frequency

    def transform(self, X):
        """
        Encode categorical features based on the learned frequencies.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with categorical features encoded based on frequencies.
        """
        X_transformed = X.copy()
        
        # Apply frequencies mapping
        for cat_col in self.categorical_features:
            X_transformed[cat_col] = X_transformed[cat_col].map(self.mapper[cat_col])
            
        return X_transformed

    def fit_transform(self, X):
        """
        Combined fit and transform operations.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with categorical features encoded based on frequencies.
        """
        self.fit(X)
        return self.transform(X)



