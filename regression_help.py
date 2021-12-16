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


# Please ignore the next two comment lines
### Scaler functions define 4 parameters, your train, validate and test sets as well a list of columns to scale, and add appropriately scaled columms to your train, validate, and test dataframes.
## Author will write more code when called upon to scale more than 8 columns, because he couldn't figure out how to loop this mess.

# Author lifted code so that new scaler functions (involving very few lines of code compared to the old scaler functions) are not limited by number of features.

### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ROBUST SCALERS /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ###

def busta_scale(train, validate, test, quant_vars):

    #define k, number of features to scale
    k = len(quant_vars)

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


### AHEM!
#actually just use this function instead
def robust_scaler(train, validate, test, quant_vars):
    
    #creation
    scaler = sklearn.preprocessing.RobustScaler()
    #fitting
    scaler.fit(train[quant_vars])

    #assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    #transforming
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    #return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler

#and when you use this function, do it like this:
#train_scaled, validate_scaled, test_scaled, scaler = insert_scaler_function_here(train, validate, test, quant_vars)


### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ MIN-MAX SCALERS /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ###



def min_max_scale(train, validate, test, quant_vars):

    #set up number of features to scale
    k = len(quant_vars)


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


### Stop! Stop... stop.
#just use this function instead (can scale more than 8 features lol)
def max_min_scaler(train, validate, test, quant_vars):

    #creation
    scaler = sklearn.preprocessing.MinMaxScaler()
    #fitting
    scaler.fit(train[quant_vars])

    #assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    #autobots, roll out
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    #return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler

#and when you use this function, do it like this:
#train_scaled, validate_scaled, test_scaled, scaler = insert_scaler_function_here(train, validate, test, quant_vars)




### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ STANDARD SCALERS /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ###




def standard_scale(train, validate, test, quant_vars):

    #set up k
    k = len(quant_vars)

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



### Freeze!
#this function is not limited by a number of features to scale
def standard_scaler(train, validate, test, quant_vars):

    #creation
    scaler = sklearn.preprocessing.StandardScaler()
    #fitting
    scaler.fit(train[quant_vars])
    
    #assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    #autobots do it
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    #return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler

#and when you use this function, do it like this:
#train_scaled, validate_scaled, test_scaled, scaler = insert_scaler_function_here(train, validate, test, quant_vars)




### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ QUANTILE SCALERS FOUND HERE /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ###




##### NORMAL QUANTILE SCALER
def quantile_norm_scale(train, validate, test, quant_vars):

    #set up number of features to scale
    k = len(quant_vars)

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


### Halt!
#this function is not limited by a number of features to scale
def quantile_norm_scaler(train, validate, test, quant_vars):

    #creation
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution = 'normal')
    #fitting
    scaler.fit(train[quant_vars])
    
    #assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    #autobots do it
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    #return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler

#and when you use this function, do it like this:
#train_scaled, validate_scaled, test_scaled, scaler = insert_scaler_function_here(train, validate, test, quant_vars)


##### UNIFORM QUANTILE SCALER
def quantile_uniform_scale(train, validate, test, quant_vars):

    #set up k
    k = len(quant_vars)

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



### Attention!
#this function is not limited by a number of features to scale
def quantile_uniform_scaler(train, validate, test, quant_vars):

    #creation
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution = 'uniform')
    #fitting
    scaler.fit(train[quant_vars])
    #assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    #autobots do it
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    #return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler

#and when you use this function, do it like this:
#train_scaled, validate_scaled, test_scaled, scaler = insert_scaler_function_here(train, validate, test, quant_vars)




### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ FEATURE SELECTION FUNCTIONS HERE /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ###



### Note: must define X_train and y_train prior to running feature selection functions
## note: also these lists are ordered backward

#X_train = predictors or features (same thing if you got the right features)
#y_train = target
#k = number of features you want

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




### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ MODEL DATA SETUP HOSTED HERE /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ###


## Notes: must have used the above scale functions (NOT THE SCALER ONES, THOSE ONES RENDER THIS PROCESS OBSOLETE) or have defined a list of the columns you scaled prior to scaling and imported new columns into your dataframes that are named: f'{column}_scaled' prior to running this function.
## Notes: If you want this function to be of use, recommend setting up X_train (afterward) with a list called scaled, such as:
    ### X_train = train[scaled]
    ### y_train = train['target_variable'] 

    ### X_validate = validate[scaled]
    ### y_validate = and so on and so forth...

    ### X_test = 

    ## OR when you fit your model onto the training data, you can pick and choose between scaled features and slice the list. you must be familar with your list lol
    ### dang this is kinda starting to look like what feature selection already does. IT IS TIME TO END IT. THIS IS AS FAR AS THIS RABBIT HOLE GOES.

    ### linear_model.fit(X_train[scaled[2]], y_train)
    ### train['y_hat'] = linear_model.predict(X_train[scaled[2]])

    ## OR I just stayed up really late coding away over nothing cuz you don't need a list if you wanna pick and choose from the list, you can access a subset of the dataframe

    ### lm2.fit(X_train[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']], y_train)
    ### validate['y_hat'] = lm2.predict(X_validate[[f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']])

## Otherwise these functions are a total waste and a buzz kill
## This list only has value if you are fitting your model onto all of the scaled features (just do it) or are adept at slicing into dataframes



### Why did I write this function?
#list_scaled defines 4 parameters, your train, validate and test sets, your list of features you scaled (quant_vars), and returns a list of the scaled columns (scaled).
def list_scaled(train, validate, test, quant_vars):
    
    #set k equal to length of list of features scaled
    k = len(quant_vars)

    #set up a chain of if-conditionals to see what k (number of features scaled or len(quant_vars)) is, and then
    if k == 1:
        #use the force
        scaled = [f'{quant_vars[0]}_scaled']

        #return modeling data list
        return scaled

    elif k == 2:
        #c'mon, luke
        scaled = [f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled']

        #return modeling data list
        return scaled

    elif k == 3:
        #let's go, Rey
        scaled = [f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled']

        #return modeling data list
        return scaled
    
    elif k == 4:
        #we're a squad
        scaled = [f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled']
        
        #return list
        return scaled
    
    elif k == 5:
        #we don't leave our mates behind
        scaled = [f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled']

        #return list
        return scaled

    elif k == 6:
        #I'm Luke Skywalker; I'm here to rescue you!
        scaled = [f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled']

        #return list
        return scaled

    elif k == 7:
        #I'm a Jedi, like my father before me
        scaled = [f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled']

        #return list
        return scaled

    #let's stop with 8 features
    elif k == 8:
        #ok im a jedi master let me freak out over my nephew possibly becoming darth vader even though I'm a resolved character in a previous trilogy
        scaled = [f'{quant_vars[0]}_scaled', f'{quant_vars[1]}_scaled', f'{quant_vars[2]}_scaled', f'{quant_vars[3]}_scaled', f'{quant_vars[4]}_scaled', f'{quant_vars[5]}_scaled', f'{quant_vars[6]}_scaled', f'{quant_vars[7]}_scaled']

        #return list
        return scaled