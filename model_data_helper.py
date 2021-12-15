import warnings
warnings.filterwarnings('ignore')
import sklearn.preprocessing

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import mason_functions as mf



#quant_vars is a list of eligible columns you wish to scale
#Note: You must load your train, validate and test samples, and you must define quant_vars before running the scaler functions.

### Scaler functions define 4 parameters, your train, validate and test sets as well as k, the number of features to scale (up to 8), and add appropriately scaled columms to your train, validate, and test dataframes.
## Author will write more code when called upon to scale more than 8 columns, because he couldn't figure out how to loop this mess.

### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ROBUST SCALER /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ###

def busta_scale(train, validate, test, quant_vars, k):

    #creation
    scaler = sklearn.preprocessing.RobustScaler()

    #fit the object
    scaler.fit(train[quant_vars])

    #set up a chain of if-conditionals to see what k (number of features to scale or len(quant_vars)) is, and then
    if k == 1:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 2:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test



    elif k == 3:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test
    

    elif k == 4:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = test_scaled
        
        #return dataframes with added columns
        return train, validate, test
    

    elif k == 5:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test


    elif k == 6:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test


    elif k == 7:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    #let's stop with 8 features
    elif k == 8:

        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test



### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ MIN-MAX SCALER /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ###



def min_max_scale(train, validate, test, quant_vars, k):

    #creation
    scaler = sklearn.preprocessing.MinMaxScaler()

    #fit the object
    scaler.fit(train[quant_vars])

    #set up a chain of if-conditionals to see what k (number of features to scale or len(quant_vars)) is, and then
    if k == 1:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 2:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 3:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test
    
    elif k == 4:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = test_scaled
        
        #return dataframes with added columns
        return train, validate, test
    
    elif k == 5:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 6:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 7:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    #let's stop with 8 features
    elif k == 8:

        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test



### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ STANDARD SCALER /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ###




def standard_scale(train, validate, test, quant_vars, k):

    #creation
    scaler = sklearn.preprocessing.StandardScaler()

    #fit the object
    scaler.fit(train[quant_vars])

    #set up a chain of if-conditionals to see what k (number of features to scale or len(quant_vars)) is, and then
    if k == 1:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 2:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 3:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test
    
    elif k == 4:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = test_scaled
        
        #return dataframes with added columns
        return train, validate, test
    
    elif k == 5:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 6:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 7:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    #let's stop with 8 features
    elif k == 8:

        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test




### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ QUANTILE SCALERS FOUND HERE /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ###




##### NORMAL QUANTILE SCALER
def quantile_norm_scale(train, validate, test, quant_vars, k):

    #creation
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution = 'normal')

    #fit the object
    scaler.fit(train[quant_vars])

    #set up a chain of if-conditionals to see what k (number of features to scale or len(quant_vars)) is, and then
    if k == 1:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 2:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 3:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test
    
    elif k == 4:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = test_scaled
        
        #return dataframes with added columns
        return train, validate, test
    
    elif k == 5:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 6:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 7:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    #let's stop with 8 features
    elif k == 8:

        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test



##### UNIFORM QUANTILE SCALER
def quantile_uniform_scale(train, validate, test, quant_vars, k):

    #creation
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution = 'uniform')

    #fit the object
    scaler.fit(train[quant_vars])

    #set up a chain of if-conditionals to see what k (number of features to scale or len(quant_vars)) is, and then
    if k == 1:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 2:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 3:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test
    
    elif k == 4:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']] = test_scaled
        
        #return dataframes with added columns
        return train, validate, test
    
    elif k == 5:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 6:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    elif k == 7:
        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test

    #let's stop with 8 features
    elif k == 8:

        #use the object
        train_scaled = scaler.transform(train[quant_vars])
        validate_scaled = scaler.transform(validate[quant_vars])
        test_scaled = scaler.transform(test[quant_vars])

        #add columns
        train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']]= train_scaled
        validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']] = validate_scaled
        test[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']] = test_scaled

        #return dataframes with added columns
        return train, validate, test






### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ FEATURE SELECTION FUNCTIONS HERE /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ###


### Note: must define X_train and y_train prior to running feature selection functions

#X_train = predictors or features (same thing if you got the right features)
#y_train = target
#k = number of features

#select_kbest defines 3 parameters, X_train (predictors), y_train (target variable) and k (number of features to spit), and returns a list of the best features my man
def select_kbest(X_train, y_train, k):

    #import feature selection tools
    from sklearn.feature_selection import SelectKBest, f_regression

    #create the selector
    f_select = SelectKBest(f_regression, k = k)

    #fit the selector
    f_select.fit(X_train, y_train)

    #create a boolean mask to show if feature was selected
    feat_mask = f_select.get_support()
    
    #create a list of the best features
    best_features = X_train.iloc[:,feat_mask].columns.to_list()

    #gimme gimme
    return best_features



#rfe defines 3 parameters, X_train (features), y_train (target variable) and k (number of features to bop), and returns a list of the best boppits m8
def rfe(X_train, y_train, k):

    #import feature selection tools
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression

    #crank it
    lm = LinearRegression()

    #pop it
    rfe = RFE(lm, k)
    
    #bop it
    rfe.fit(X_train, y_train)  
    
    #twist it
    feat_mask = rfe.support_
    
    #pull it 
    best_rfe = X_train.iloc[:,feat_mask].columns.tolist()
    
    #bop it
    return best_rfe