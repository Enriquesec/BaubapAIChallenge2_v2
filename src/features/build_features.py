import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy import stats as st
import category_encoders as ce

import sys
sys.path.append("../../")

from BaubapAIChallenge2.src.features.utils import *

def preprocess_data(X: pd.DataFrame, X_val: pd.DataFrame, y, numeric_strategy='mean', categorical_strategy='most_frequent',
                   threshold=0.85, cat_transformer_max_unique_values=20, cardinality_max_unique_values=2000,
                   n_components=40, cat_encoder_strategy='woee'):
    """
    Preprocesses input DataFrames for modeling.

    Parameters:
    - X (pd.DataFrame): Training data DataFrame.
    - X_val (pd.DataFrame): Validation data DataFrame.
    - numeric_strategy (str): Strategy for filling missing values in numeric columns (default is 'mean').
    - categorical_strategy (str): Strategy for filling missing values in categorical columns (default is 'most_frequent').
    - threshold (float): Threshold for dropping highly correlated features (default is 0.85).
    - cat_transformer_max_unique_values (int): Maximum unique values allowed for categorical transformer (default is 20).
    - cardinality_max_unique_values (int): Maximum unique values allowed for constants and high-cardinality dropper (default is 2000).
    - n_components (int): Number of components for PCA dimensionality reduction (default is 40).
    - cat_encoder_strategy (str): Strategy for categorical encoding ('woee', 'freq', or 'ohe', default is 'woee').

    Returns:
    - X_transformed (pd.DataFrame): Preprocessed training data.
    - X_val_transformed (pd.DataFrame): Preprocessed validation data.
    """
    # Step 0: Copy Values to transform
    X_transformed = X.copy()
    X_val_transformed = X_val.copy()
    
    # Step 1: Fill missing values
    filler = MissingValuesFiller(numeric_strategy=numeric_strategy, categorical_strategy=categorical_strategy)
    X_transformed = filler.fit_transform(X_transformed)
    X_val_transformed = filler.transform(X_val_transformed)
    
    # Step 2: Reduce dimensionality due to high correlation
    correlation_reducer = DropHighCorrelationFeatures(threshold=threshold)
    X_transformed = correlation_reducer.fit_transform(X_transformed)
    X_val_transformed = correlation_reducer.transform(X_val_transformed)
    
    # Step 3: Convert features to categorical
    categorical_transformer = CategoricalTransformer(max_unique_values=cat_transformer_max_unique_values).fit(X_transformed)
    X_transformed = categorical_transformer.transform(X_transformed)
    X_val_transformed = categorical_transformer.transform(X_val_transformed)
    
    # Step 4: Drop constants and high-cardinality features (almost identifiers)
    dropper = ConstantsAndHighCardinalityDropper(max_unique_values=cardinality_max_unique_values)
    X_transformed = dropper.fit_transform(X_transformed)
    X_val_transformed = dropper.transform(X_val_transformed)
    
    # Step 5: Split the DataFrame into numeric and categorical parts
    splitter = SplitterByType()
    X_transformed_numeric, X_transformed_categorical = splitter.fit_transform(X_transformed)
    X_val_transformed_numeric, X_val_transformed_categorical = splitter.transform(X_val_transformed)
    
    ## Excluding (kike's variables) different distributed variables with hypothesis testing .drop['f1',f91']
    incosistencias_columns = ['Feature_11', 'Feature_71', 'Feature_88', 'Feature_90',
       'Feature_143', 'Feature_156', 'Feature_176', 'Feature_194',
       'Feature_200', 'Feature_216', 'Feature_237', 'Feature_268',
       'Feature_273', 'Feature_319', 'Feature_339', 'Feature_386',
       'Feature_391', 'Feature_402', 'Feature_407', 'Feature_410',
       'Feature_417', 'Feature_421', 'Feature_422', 'Feature_427',
       'Feature_431', 'Feature_441', 'Feature_471', 'Feature_477',
       'Feature_492', 'Feature_532']
    col_new = set(X_transformed_numeric)-set(incosistencias_columns)
    X_transformed_numeric = X_transformed_numeric[col_new]
    X_val_transformed_numeric = X_val_transformed_numeric[col_new]
    
    # Step 6: Reduce dimensionality of numeric features
    pca = PCA(n_components)
    # X_transformed_numeric = pd.DataFrame(pca.fit_transform(X_transformed_numeric))
    # X_val_transformed_numeric = pd.DataFrame(pca.transform(X_val_transformed_numeric))
    # print(f'Explained Variance with {n_components} components: {pca.explained_variance_ratio_.sum()}')
    
    # Step 7: Categorical Encoding
    if cat_encoder_strategy == 'woee':
        # Step 7.1 (Optional): Apply WOE Encoding
        X_columns = X_transformed_categorical.columns
        woe_encoder = ce.WOEEncoder(cols=X_columns)
        X_transformed_categorical = woe_encoder.fit_transform(X_transformed_categorical, y)
        X_val_transformed_categorical = woe_encoder.transform(X_val_transformed_categorical)
    elif cat_encoder_strategy == 'freq':
        # Step 7.2 (Optional): Apply self-defined Frequency Encoding
        X_columns = X_transformed_categorical.columns
        freq_encoder = FrequencyEncoder()
        X_transformed_categorical = freq_encoder.fit_transform(X_transformed_categorical)
        X_val_transformed_categorical = freq_encoder.transform(X_val_transformed_categorical)
    else:
        # Step 7.3 (Optional): Apply One Hot Encoding
        X_columns = X_transformed_categorical.columns
        ohe_encoder = ce.OneHotEncoder()
        X_transformed_categorical = ohe_encoder.fit_transform(X_transformed_categorical)
        X_val_transformed_categorical = ohe_encoder.transform(X_val_transformed_categorical)
        
    # Step 8: Mixing steps 6 & 7 to prepare data for modeling
    X_transformed = pd.concat([X_transformed_numeric, X_transformed_categorical], axis=1)
    X_val_transformed = pd.concat([X_val_transformed_numeric, X_val_transformed_categorical], axis=1)
    
    return X_transformed, X_val_transformed

