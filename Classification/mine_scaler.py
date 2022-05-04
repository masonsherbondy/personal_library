import sklearn.preprocessing

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# INDEX #
## ROBUST SCALER ##
## MIN-MAX SCALER ##
## STANDARD SCALER ##
## QUANTILE SCALER ##




## ROBUST SCALER ##
def robust_scaler(train, validate, test, quant_vars):

    '''
    This function takes in train, validate and test sets and a list of numeric features to scale (quant_vars). It then scales these features using Sci-Kit Learn's 
    robust scaler. First, it creates the scaler. Then, it fits the scaler to the numeric features of the train set. It then creates 3 new dataframes out of the 
    numeric features of the 3 subsets. It then scales (transforms) the data and returns the scaled subsets as well as the scaler.
    '''
    
    # creation
    scaler = sklearn.preprocessing.RobustScaler()
    # fitting
    scaler.fit(train[quant_vars])

    # assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    # transforming
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    # return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler


## MIN-MAX SCALER ##
def max_min_scaler(train, validate, test, quant_vars):

    '''
    This function takes in train, validate and test sets and a list of numeric features to scale (quant_vars). It then scales these features using Sci-Kit Learn's 
    min-max scaler. First, it creates the scaler. Then, it fits the scaler to the numeric features of the train set. It then creates 3 new dataframes out of the 
    numeric features of the 3 subsets. It then transforms the data and returns the transformed subsets as well as the Transformer.
    '''

    # creation
    scaler = sklearn.preprocessing.MinMaxScaler()
    # fitting
    scaler.fit(train[quant_vars])

    # assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    # autobots, roll out
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    # return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler


## STANDARD SCALER ##
def standard_scaler(train, validate, test, quant_vars):

    '''
    This function takes in train, validate and test sets and a list of numeric features to scale (quant_vars). It then scales these features using Sci-Kit Learn's 
    standard scaler. First, it creates the scaler. Then, it fits the scaler to the numeric features of the train set. It then creates 3 new dataframes out of the 
    numeric features of the 3 subsets. It then scales (transforms) the data and returns the scaled subsets as well as the scaler.
    '''
    
    # creation
    scaler = sklearn.preprocessing.StandardScaler()
    # fitting
    scaler.fit(train[quant_vars])
    
    # assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    # autobots do it
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    # return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler

## QUANTILE SCALER ##
def quantile_norm_scaler(train, validate, test, quant_vars):

    '''
    This function takes in train, validate and test sets and a list of numeric features to scale (quant_vars). It then scales these features using Sci-Kit Learn's 
    quantile scaler. First, it creates the scaler. Then, it fits the scaler to the numeric features of the train set. It then creates 3 new dataframes out of the 
    numeric features of the 3 subsets. It then scales (transforms) the data and returns the scaled subsets as well as the scaler.
    '''
    
    # creation
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution = 'normal')
    # fitting
    scaler.fit(train[quant_vars])
    
    # assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    # autobots do it
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    # return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler

## QUANTILE SCALER (UNIFORM) ##
def quantile_uniform_scaler(train, validate, test, quant_vars):

    '''
    This function takes in train, validate and test sets and a list of numeric features to scale (quant_vars). It then scales these features using Sci-Kit Learn's 
    quantile scaler. First, it creates the scaler. Then, it fits the scaler to the numeric features of the train set. It then creates 3 new dataframes out of the 
    numeric features of the 3 subsets. It then scales (transforms) the data and returns the scaled subsets as well as the scaler.
    '''
    
    # creation
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution = 'uniform')

    # fitting
    scaler.fit(train[quant_vars])
    
    # assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    # autobots do it
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    # return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler