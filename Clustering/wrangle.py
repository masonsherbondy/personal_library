import warnings
warnings.filterwarnings('ignore')
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import mason_functions as mf






def calculate_column_nulls(df):
    
    #set an empty list
    output = []
    
    #gather columns
    df_columns = df.columns.to_list()
    
    #commence for loop
    for column in df_columns:
        
        #assign variable to number of rows that have null values
        missing = df[column].isnull().sum()
        
        #assign variable to ratio of rows with null values to overall rows in column
        ratio = df[column].isnull().sum() / len(df)
    
        #assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100, 2)}%'
                 }
        #add dictonaries to list
        output.append(r_dict)
        
    #design frame
    column_nulls = pd.DataFrame(output, index = df_columns)
    
    #return the dataframe
    return column_nulls


    def calculate_row_nulls(df):
    
    #create an empty list
    output = []
    
    #gather values in a series
    row_nulls = df.isnull().sum(axis = 1)
    
    #commence 4 loop
    for n in range(len(row_nulls)):
        
        #assign variable to nulls
        missing = row_nulls[n]
        
        #assign variable to ratio
        ratio = missing / len(df.columns)
        
        #assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100)}%'
                 }
        #add dictonaries to list
        output.append(r_dict)
        
    #design frame
    row_summary = pd.DataFrame(output, index = df.index)
    
    #return the dataframe
    return row_summary