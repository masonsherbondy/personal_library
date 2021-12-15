#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import acquire




#------------------------------------- IRIS DATA -------------------------------------#

                              
def prep_iris():
    '''
    This function acquires the iris data through my acquire module, and it drops any seemingly unhelpful
    columns and one-hot encodes my target variable for modeling purposes.
    '''
        
    iris_df = acquire.get_iris_data()
    
    #I no longer need these columns
    iris_df.drop(columns = ['species_id', 'measurement_id'], inplace = True)
    
    #species_name is lenghthily superfluous, just like this comment
    iris_df.rename(columns = {'species_name': 'species'}, inplace = True)
    
    #one-hot encode my target variable
    iris_dummy_df = pd.get_dummies(iris_df['species'], drop_first = False)
    
    #concatenate the resulting dataframes
    iris = pd.concat([iris_df, iris_dummy_df], axis = 1)
    
    #return the end result
    return iris


def split_iris(df):
    '''
    Takes in my iris dataframe and returns the train, validate, and test subset dataframes
    '''
    
    #make my training set and my test set, stratify the target variable 'species'
    train, test = train_test_split(df, test_size = .2, random_state = 421, stratify = df.species)
    
    #make my validate set from my train set, stratify the target variable
    train, validate = train_test_split(train, test_size = .3, random_state = 421, stratify = train.species)
    
    #get back the 3 different data sets
    return train, validate, test


def prep_iris_data():
    '''
    This function acquires the iris data through my acquire module, and it drops any seemingly unhelpful
    columns and one-hot encodes my target variable for modeling purposes.
    '''
        
    #prep iris data
    iris = prep_iris()
    
    #split iris data
    train, validate, test = split_iris(iris)
    
    #return the ensuing data frames
    return train, validate, test
    


#------------------------------------- TITANIC DATA -------------------------------------#


def prep_titanic():
    '''
    This function acquires the Titanic data through my acquire module, and it drops any seemingly unhelpful
    columns and one-hot encodes my categorical variables for modeling purposes.
    '''
    #acquire the data
    titanic_df = acquire.get_titanic_data()
    
    #drop duplicates if any
    titanic_df = titanic_df.drop_duplicates()
    
    #passenger_id extra, can be index
    titanic_df = titanic_df.set_index('passenger_id')
    
    #drop superfluous columns
    titanic_df = titanic_df.drop(columns = ['embarked', 'deck', 'pclass'])
    
    #fill NA values for 'embark_town'
    titanic_df['embark_town'] = titanic_df.embark_town.fillna(value = 'Southampton')
    
    #get dummies
    dummy_df = pd.get_dummies(titanic_df[['class', 'sex', 'embark_town']], dummy_na = False, drop_first = False)
    
    #concatenate resulting dataframes
    titanic = pd.concat([titanic_df, dummy_df], axis = 1)
    
    #drop more columns
    titanic = titanic.drop(columns = ['sex', 'embark_town', 'class', 'sex_male'])
    
    #last bit of cleaning
    titanic = titanic.rename(columns = {
        'class_First': 'first_class',
        'class_Second': 'second_class',
        'class_Third': 'third_class',
        'sex_female': 'female'
    })
    
    #gimme dat
    return titanic


def split_titanic(df):
    '''
    Takes in a titanic dataset dataframe and returns the train, validate, and test subset dataframes
    '''
    
    train, test = train_test_split(df, test_size = .2, random_state = 421, stratify = df.survived)
    train, validate = train_test_split(train, test_size = .3, random_state = 421, stratify = train.survived)
    return train, validate, test





def prep_titanic_data():
    '''
    This function acquires the titanic data set, preps the data, and returns my train, validate, and test dataframes.
    '''
    
    #prepare the data
    titanic = prep_titanic()
    
    #split the data
    train, validate, test = split_titanic(titanic)
    
    #yes
    return train, validate, test


def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test





def impute_mode(train, validate, test):
    '''
    uses train to identify the best value to replace nulls in embark_town with
    '''
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    train[['embark_town']] = imputer.fit_transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    return train, validate, test



#------------------------------------- TELCO DATA -------------------------------------#


def prep_telco():
    '''
    This function acquires the telco data through my acquire module, and it drops any seemingly unhelpful
    columns and one-hot encodes my categorical variables for modeling purposes.
    '''
    telco_df = acquire.get_telco_data()
    
    #change total_charges to an appropriate data type
    telco_df.total_charges = telco_df.total_charges.str.strip()
    telco_df = telco_df[telco_df.total_charges != '']
    telco_df.total_charges = telco_df.total_charges.astype(float)
    
    #customer_id is not a feature, but it can be an index
    telco_df = telco_df.set_index('customer_id')
    
    #we don't need these extra columns
    telco_df = telco_df.drop(columns = ['contract_type_id', 'internet_service_type_id', 'payment_type_id'])
    
    #get dummies m8
    dummy_df = pd.get_dummies(telco_df[[
    'multiple_lines',
    'online_security',
    'online_backup',
    'device_protection',
    'tech_support',
    'streaming_tv',
    'streaming_movies',
    'gender',
    'contract_type',
    'internet_service_type',
    'payment_type']], dummy_na = False, drop_first = False)
    
    #drop firsts (extra work here could've been avoided by setting drop_first kwg to True, but I'm picky)
    dummy_df = dummy_df.drop(columns = [
        'multiple_lines_No',
        'online_security_No',
        'online_backup_No',
        'device_protection_No',
        'tech_support_No',
        'streaming_tv_No',
        'streaming_movies_No',
        'gender_Male'
    ])
    
    #drop firsts in original dataframe
    telco_df = telco_df.drop(columns = [
        'multiple_lines',
        'online_security',
        'online_backup',
        'device_protection',
        'tech_support',
        'streaming_tv',
        'streaming_movies',
        'gender',
        'contract_type',
        'internet_service_type',
        'payment_type'
    ])
    
    #make these column names easier for myself to work with
    dummy_df.columns = dummy_df.columns.str.replace(' ', '_').str.lower().str.replace('_yes', '')
    dummy_df = dummy_df.rename(columns = {
        'gender_female': 'female',
        'contract_type_month-to-month': 'contract_m2m',
        'contract_type_one_year': 'contract_one_year',
        'contract_type_two_year': 'contract_two_year',
        'internet_service_type_dsl': 'DSL_internet',
        'internet_service_type_fiber_optic': 'Fiber_optic_internet',
        'internet_service_type_none': 'no_internet',
        'payment_type_bank_transfer_(automatic)': 'bank_transfer_auto_payment',
        'payment_type_credit_card_(automatic)': 'credit_card_auto_payment',
        'payment_type_electronic_check': 'electronic_check_payment',
        'payment_type_mailed_check': 'mailed_check_payment'
    })
    
    #more one-hot encoding    
    telco_df = telco_df.replace('Yes', 1).replace('No', 0)
    
    #concatenate the resulting dataframes    
    telco = pd.concat([telco_df, dummy_df], axis = 1)
    
    #last bit
    telco = telco.drop(columns = [
        'multiple_lines_no_phone_service',
        'online_security_no_internet_service',
        'online_backup_no_internet_service',
        'device_protection_no_internet_service',
        'tech_support_no_internet_service',
        'streaming_tv_no_internet_service',
        'streaming_movies_no_internet_service'
    ])
    
    #return the end of prep
    return telco


def split_telco(df):
    '''
    Takes in the telco dataset and returns the train, validate, and test subset dataframes.
    '''
    
    #get my training and test data sets defined
    train, test = train_test_split(df, test_size = .2, random_state = 421, stratify = df.churn)
    
    #get my validate set from the training set
    train, validate = train_test_split(train, test_size = .3, random_state = 421, stratify = train.churn)
    
    #ehhh? gimme dattt
    return train, validate, test


def prep_telco_data():
    '''
    This function acquires the telco data through my acquire module, preps the data, and returns my train, validate and test
    training sets.
    '''
    #prep the data
    telco = prep_telco()
    
    #split the data
    train, validate, test = split_telco(telco)
    
    #I want this buff
    return train, validate, test