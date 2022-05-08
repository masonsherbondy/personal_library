# This is the way
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

## INDEX
### NULL FUNCTIONS
### OUTLIER FUNCTIONS
### SPLITTER FUNCTIONS
### EXPLORE FUNCTIONS

### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NULL FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ###



def calculate_column_nulls(df):

    '''
    This function  defines one parameter, a dataframe, and returns a dataframe that holds data regarding null values and ratios (pertaining to the whole column) 
    in the original frame.
    '''   
    
    output = []    # set an empty list
    df_columns = df.columns.to_list()   # gather columns

    for column in df_columns:   # commence for-loop
        missing = df[column].isnull().sum()    # assign variable to number of rows that have null values
        ratio = missing / len(df)   # assign variable to ratio of rows with null values to overall rows in column
        # assign a dictionary for your dataframe to accept       
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100, 2)}%'
                 }
        output.append(r_dict)   # add dictonaries to list

    column_nulls = pd.DataFrame(output, index = df_columns)    # design frame
    column_nulls = column_nulls.sort_values('nulls', ascending = False)    # sort

    return column_nulls

def calculate_row_nulls(df):

    '''
    This function  defines one parameter, a dataframe, and returns a dataframe that holds data regarding null values and ratios (pertaining to the whole row) 
    in the original frame.
    '''   
    
    output = []    # create an empty list
    nulls = df.isnull().sum(axis = 1)   # gather values in a series
    for n in range(len(nulls)):    # commence 4 loop
        missing = nulls[n]     # assign variable to nulls
        ratio = missing / len(df.columns)   # assign variable to ratio
        # assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100)}%'
                 }
        output.append(r_dict)   # add dictonaries to list

    row_nulls = pd.DataFrame(output, index = df.index)    # design frame
    row_nulls = row_nulls.sort_values('nulls', ascending = False)   # sort

    return row_nulls

def handle_nulls(df, cr, rr):

    '''
    This function defines 3 parameters, the dataframe you want to calculate nulls for (df), the column ratio (cr) and the row ratio (rr),
    ratios that define the threshold of having too many nulls, and returns your dataframe with columns and rows dropped if they are above their respective threshold ratios.
    Note: This function calculates the ratio of nulls missing for rows AFTER it drops the columns with null ratios above the cr threshold.
    TL; DR: This function handles nulls for dataframes.
    '''

    ## DROP COLUMN NULLS ##
    output = []    # set an empty list    
    df_columns = df.columns.to_list()    # gather columns
    for column in df_columns:    # commence for-loop
        missing = df[column].isnull().sum()    # assign variable to number of rows that have null values
        ratio = missing / len(df)    # assign variable to ratio of rows with null values to overall rows in column
        # assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100, 2)}%'
                 }
        output.append(r_dict)    # add dictonaries to list
        
    column_nulls = pd.DataFrame(output, index = df_columns)    # design frame
    null_and_void = []    # set a list of columns to drop
    
    for n in range(len(column_nulls)):    # commence 4-loop
        if column_nulls.iloc[n].null_ratio > cr:     # set up conditional to see if the ratio of nulls to total columns is over the entered column ratio (.cr)
            null_and_void.append(column_nulls.index[n])    # add columns over the threshold with nulls to the list of columns to drop
    
    df = df.drop(columns = null_and_void)    # drop the columns

    ## DROP ROW NULLS ##
    output = []    # create another list
    nulls = df.isnull().sum(axis = 1)    # gather values in a series
    
    for n in range(len(nulls)):    # commence 4 loop
        missing = nulls[n]    # assign variable to nulls
        ratio = missing / len(df.columns)    # assign variable to ratio
        # assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100)}%'
                 }
        output.append(r_dict)    # add dictonaries to list
    
    row_nulls = pd.DataFrame(output, index = df.index)    # design frame
    ice_em = []    # set an empty index of rows to drop
    for n in range(len(row_nulls)):    # commence loop
        if row_nulls.iloc[n].null_ratio > rr:    # set up conditional to see if the ratio of nulls to total columns is over the entered row ratio (.rr)
            ice_em.append(row_nulls.index[n])    # add rows to index

    df = df.drop(index = ice_em)    # drop rows where the percentage of nulls is over the threshold
    
    return df    # return the dataframe with preferred drop parameters



### &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& OUTLIER FUNCTIONS &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& ###



def get_lower_and_upper_bounds(series, multiplier = 1.5):

    '''
    This function defines two paramters, a series and a default multiplier (int or float will do), and returns 
    the upper and lower bounds according to the multipler and series range.
    '''
    
    q1, q3 = series.quantile([.25, .75])    # get quartiles
    iqr = q3 - q1    # calculate inter-quartile range
    inner_lower_fence = q1 - multiplier * iqr    # set lower fence
    inner_upper_fence = q3 + multiplier * iqr    # set upper fence

    return inner_lower_fence, inner_upper_fence  # return lower bound and then upper bound

def get_all_the_bounds(df, quant_vars, multiplier = 1.5):
    
    '''
    This function defines 3 paramaters, a dataframe, a list of variables, and a default multiplier.
    It returns the upper and lower bounds for each variable in the list according to the multipler
    and range of the variable listed.
    '''
    
    bounds = pd.DataFrame()   # set an frame to fill
    for var in quant_vars:     # loop through numeric varumns
        q1, q3 = df[var].quantile([.25, .75])    # get quartiles
        iqr = q3 - q1    # calculate inter-quartile range
        inner_lower_fence = q1 - multiplier * iqr    # set lower fence
        inner_upper_fence = q3 + multiplier * iqr    # set upper fence
        outlier_dict = {'lower_bound': inner_lower_fence,     # add values to dictionary
                        'upper_bound': inner_upper_fence,
                        'feature': var
                       }
        
        bounds = bounds.append(outlier_dict, ignore_index = True)    # design dataframe of bounds
    
    bounds = bounds.set_index('feature')
    
    return bounds    # return bounds in an easy-on-the-eyes dataframe

def get_out_of_bounds(df, quant_vars, multiplier = 1.5):
    
    '''
    This function defines 3 paramaters, a dataframe, a list of variables, and a default multiplier.
    It returns all of the outliers present in the specified dataframe in a new dataframe
    (outlier classification dependent on set/ default multiplier).
    '''
    
    outliers = pd.DataFrame()
    
    for var in quant_vars:     # loop through numeric varumns
        q1, q3 = df[var].quantile([.25, .75])    # get quartiles
        iqr = q3 - q1    # calculate inter-quartile range
        inner_lower_fence = q1 - multiplier * iqr    # set lower fence
        inner_upper_fence = q3 + multiplier * iqr    # set upper fence
        outliers = outliers.append(df[df[var] < inner_lower_fence])    # add lower outliers to df
        outliers = outliers.append(df[df[var] > inner_upper_fence])    # add upper outliers to df

    return outliers    # return outliers in an easy-on-the-eyes dataframe

def get_below_bounds(df, quant_vars, multiplier = 1.5):

    '''
    This function defines 3 paramaters, a dataframe, a list of variables, and a default multiplier.
    It returns all of the outliers below their respective lower bounds present in the specified dataframe
    in a new dataframe (outlier classification dependent on set/ default multiplier).
    '''
    
    below_ideal = pd.DataFrame()
    for var in quant_vars:
        q1, q3 = df[var].quantile([.25, .75])    # get quartiles
        iqr = q3 - q1    # calculate inter-quartile range
        lower_fence = q1 - multiplier * iqr    # set lower fence
        below_ideal = below_ideal.append(df[df[var] < lower_fence])    # add outliers to dataframe
    
    return below_ideal    # return lower outliers accumulated in a dataframe

def get_above_bounds(df, quant_vars, multiplier = 1.5):
    
    '''
    This function defines 3 paramaters, a dataframe, a list of variables, and a default multiplier.
    It returns all of the outliers above their respective upper bounds present in the specified dataframe
    in a new dataframe (outlier classification dependent on set/ default multiplier).
    '''

    above_bounds = pd.DataFrame()
    
    for var in quant_vars:
        q1, q3 = df[var].quantile([.25, .75])    # get quartiles
        iqr = q3 - q1    # calculate inter-quartile range
        upper_fence = q3 + multiplier * iqr    # set lower fence
        above_bounds = above_bounds.append(df[df[var] > upper_fence])    # add outliers to dataframe
    
    return above_bounds    # return upper outliers accumulated in a dataframe



### ()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()() SPLITTER FUNCTIONS ()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()() ###



# reference for splitting data (classification method)
# split_data_strat defines two parameters, a clean dataframe (df) and my target variable (target), and returns my train, validate and test sets with the target variable stratified amongst them, whatever that means.
def class_split_data(df, target):

    '''
    Takes in a dataset and returns the train, validate, and test subset dataframes.
    Dataframe size for my test set is .2 or 20% of the original data. 
    Validate data is 30% of my training set, which is 24% of the original data. 
    Training data is 70% of my original training set, which is 56% total of the original data.
    Stratifies by the specified target variable.
    '''

    # import splitter
    from sklearn.model_selection import train_test_split

    # get my training and test data sets defined; stratify my target variable
    train, test = train_test_split(df, test_size = .2, random_state = 421, stratify = df[target])
    # get my validate set from the training set; stratify my target variable again
    train, validate = train_test_split(train, test_size = .3, random_state = 421, stratify = train[target])
    
    # return the 3 dataframes
    return train, validate, test

def one_split(df, target):

    # import splitter
    from sklearn.model_selection import train_test_split

    # assign variables to split frames, stratify the split with specified target variable
    train, validate = train_test_split(df, test_size = .3, random_state = 421, stratify = df[target])

    # return the two frames
    return train, validate



### 69696969696969696969696969696969696969696969696969696969696969 EXPLORE FUNCTIONS HERE 69696969696969696969696969696969696969696969696969696969696969 ###


def get_values_and_counts(df):

    '''
    This function accepts a dataframe as input, and then returns a list of columns and unique values paired with the column, and it also returns a list of 
    values and counts for each column.
    '''

    # print
    print('Values and Counts')
    print('-----------------')
    # loop through columns to print number of unique values that each feature has and print out values and counts for each feature
    for col in df.columns:
        print(f'Column:')
        print(f'{col}')
        print(f'---')
        print(f'Unique values:')
        print(f'{df[col].nunique()}')
        print(f'---')
        print(f'Counts:')
        print(df[col].value_counts(dropna = False).sort_values(ascending = False))
        print('---')
        print('Column data type:')
        print(df[col].dtype)
        print('---')
        print(f'=================================================')
        print(f'              ')


def list_floats(df):

    '''
    This function accepts a dataframe as input, and returns a list of columns with the dtype 'float64'.
    '''

    # set an empty list to fill
    quant_vars = []
    # commence loop through columns
    for column in df.columns:
        # set if-conditional to see if the column is a float 
        if df[column].dtype == 'float64':
            # if it is, add this column to the list to fill
            quant_vars.append(column)
    
    return quant_vars

# distribution defines two parameters, both a dataframe and feature to accept, and plots a histogram with chart mods for clarity
def distribution(df, feature):

    '''
    This function plots a histogram from specified df for specified feature.
    '''

    plt.figure(figsize = (7, 4))    # create figure
    df[feature].hist(color = 'darkslateblue', edgecolor = 'indigo')    # plot histogram of feature
    plt.tight_layout()    # clear it up
    plt.xticks(rotation = 45, size = 11)    # rotate x-axis label ticks 45 degrees, increase size to 11
    plt.yticks(size = 13)    # increasee y-axis label ticks to size 13
    f_feature = feature.replace('_', ' ').capitalize()    # re-format string for title
    f_feature_2 = feature.replace('_', ' ')    # x-axis string
    plt.title(f'Distribution of {f_feature}', size = 13)    # title
    plt.ylabel('Frequency', size = 11)
    plt.xlabel(f_feature_2)
    plt.grid(False)

def boxplot(df, feature):

    '''
    This function plots a boxplot from specified df for specified feature.
    '''

    title_string = feature.capitalize().replace('_', ' ')    # create title string
    plt.title(f'Distribution of {title_string}')    # title
    plt.boxplot(df[feature])   # display boxplot for column
    plt.ylabel(feature)     # label y-axis
    plt.grid(True)      # show gridlines
    plt.tight_layout();    # clean

def distribution_dos(df1, df2, feature):

    '''
    This function takes in 3 parameters, two dataframes (df1 & df2) that represent two independent sub-populations, and a feature to compare
    their respective distributions of side by side. It then plots the distributions as planned. Currently outfitted for the Spaceship Titanic data set.
    '''

    plt.figure(figsize = (8, 5))     # create figure
    plt.hist([df1[feature], df2[feature]],      # plot distributions side by side
            label = ['Transported', 'Not-transported'],
            color = ['indigo', 'mediumvioletred']
            )
    plt.legend()    # include legend
    f_feature = feature.replace('_', ' ').capitalize()    # re-format string
    plt.xlabel(f_feature, size = 13)     # x-axis label 
    plt.ylabel('Frequency', size = 13);     # y-axis label

def distributions_grid(df, quant_vars):

    '''
    This function creates a nice sized figure, enumerates the list of features passed into the function, creates a grid of subplots,
    and then charts histograms for features in the list onto the subplots.
    '''

    plt.figure(figsize = (20, 11))    # create figure
    n = len(quant_vars)
    if n % 2 != 0:
        for i, cat in enumerate(quant_vars):    # loop through enumerated list
            plot_number = i + 1    # i starts at 0, but plot nos should start at 1
            plt.subplot((n // 2) + 1, 2, plot_number)    # create subplot
            title_string = cat.capitalize().replace('_', ' ')    # create title string
            plt.title(title_string)    # title
            df[cat].hist(color = 'indigo', edgecolor = 'black', bins = 17)    # display histogram for column
            plt.grid(False)    # rid grid-lines
            plt.ylabel('Frequency')    # label y-axis
            plt.tight_layout();    # clean

    elif n % 2 == 0:
        for i, cat in enumerate(quant_vars):    # loop through enumerated list
            plot_number = i + 1    # i starts at 0, but plot nos should start at 1
            plt.subplot(n / 2, 2, plot_number)    # create subplot
            title_string = cat.capitalize().replace('_', ' ')    # create title string
            plt.title(title_string)    # title
            df[cat].hist(color = 'indigo', edgecolor = 'black', bins = 17)    # display histogram for column
            plt.grid(False)    # rid grid-lines
            plt.ylabel('Frequency')    # label y-axis
            plt.tight_layout();    # clean

def boxplot_grid(df, quant_vars):

    '''
    This function creates a nice sized figure, enumerates the list of features passed into the function, creates a grid of subplots,
    and then charts histograms for features in the list onto the subplots.
    '''

    plt.figure(figsize = (20, 11))    # create figure
    n = len(quant_vars)
    if n % 2 != 0:
        for i, cat in enumerate(quant_vars):    # loop through enumerated list
            plot_number = i + 1    # i starts at 0, but plot nos should start at 1
            plt.subplot((n // 2) + 1, 2, plot_number)    # create subplot
            title_string = cat.capitalize().replace('_', ' ')    # create title string
            plt.title(title_string)    # title
            plt.boxplot(df[cat])   # display boxplot for column
            plt.ylabel(cat, size = 18)     # label y-axis
            plt.yticks(size = 16)   # increase size on y-axis ticks
            plt.grid(True)      # show gridlines
            plt.tight_layout();    # clean

    elif n % 2 == 0:
        for i, cat in enumerate(quant_vars):    # loop through enumerated list
            plot_number = i + 1    # i starts at 0, but plot nos should start at 1
            plt.subplot(n / 2, 2, plot_number)    # create subplot
            title_string = cat.capitalize().replace('_', ' ')    # create title string
            plt.title(title_string)    # title
            plt.boxplot(df[cat])   # display boxplot for column
            plt.ylabel(cat, size = 18)     # label y-axis
            plt.yticks(size = 16)   # increase size on y-axis ticks
            plt.grid(True)      # show gridlines
            plt.tight_layout();    # clean


def compare_target_rates(df, target, x):

    '''
    This function accepts a dataframe, a target variable as a string and an x (categorical variable) to visually compare target rates across the different 
    unique values of feature.
    '''

    target_string = target.capitalize()
    sns.barplot(x = x, y = target, data = df, palette = 'Oranges')    # compare target rates across category x
    plt.axhline(df[target].mean(), label = f'Overall {target} rate')    # show overall target rate
    plt.legend()    # label horizontal plot line
    plt.grid(False)    # good-bye, gridlines


def fixed_dist_plot(df, quant_vars):

    '''
    This function creates a nice sized figure, enumerates the list of features passed into the function, creates a grid of subplots,
    and then charts histograms for features in the list onto the subplots.
    '''

    plt.figure(figsize = (20, 11))    # create figure

    for i, cat in enumerate(quant_vars):    # loop through enumerated list
        plot_number = i + 1     # i starts at 0, but plot nos should start at 1
        plt.subplot(2, 2, plot_number)    # create subplot
        plt.title(cat)    # title        
        df[cat].hist(color = 'indigo', edgecolor = 'black', bins = 20)    # display histogram for column
        plt.tight_layout();    # clean


## CUSTOMIZED STATS RETURNS ##
#### Chi Squared
#### Independent Populations T-Test
#### Levene's
#### Bartlett's

#### Chi Squared
def return_chi2(observed, alpha = .05):

    '''
    This function defines one parameter, an observed cross-tabulation, runs the stats.chi2_contingency function and returns the test results in a readable format.
    '''    

    # run the test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    # print the rest
    print('Observed')
    print('--------')
    print(observed.values)
    print('===================')
    print('Expected')
    print('--------')
    print(expected.astype(int))
    print('===================')
    print(f'Degrees of Freedom: {degf}')
    print('===================')
    print('Chi^2 and P')
    print('-----------')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p = {p:.4f}')
    print('-')
    if p < alpha:
        print('Reject null hypothesis.')
    else:
        print('Fail to reject null hypothesis.')

#### Independent Populations T-Test
def return_ttest_ind(subset1, subset2, continuous_feature, equal_var = True, alternative = 'two-sided', alpha = .05):

    '''
    This function accepts two subset dataframes, a continous feature to compare the mean of, a default argument of homoscedasticity, a default argument of 
    a two-sided t-test, and a default alpha of .05. It returns the t-statistic as well as the p-value, and it prods the user to reject or fail to reject the 
    null hypothesis.
    '''

    # run the test
    t_stat, p = stats.ttest_ind(subset1[continuous_feature], subset2[continuous_feature], equal_var = equal_var, alternative = alternative)

    # print the rest
    print(f'T-Statistic: {t_stat}')
    print(f'p-value: {p}')
    print(f'---')
    if p < alpha:
        print(f'Reject null hypothesis with {round((1 - p) * 100)}% confidence.')
    else:
        print('Fail to reject null hypothesis.')

#### Levene's (when the distribution has departed from normal like not a bell curve yo)
def return_levene(subset1, subset2, continuous_feature):

    '''
    This function accepts two subset dataframes and a feature to compare variances across, prints their variances, and prints the statistic and p-value from
    Levene's test and suggests whether the user should reject the null hypothesis that the variances are the same or to fail to reject this null hypothesis.
    '''

    # run the test
    stat, p = stats.levene(subset1[continuous_feature], subset2[continuous_feature])
    
    # print variances
    print(f'Variances:{subset1[continuous_feature].var(), subset2[continuous_feature].var()}')

    # print suggested homoscedastic verdict
    print("Levene's Test")
    print("-------------")
    print(f'Statistic: {stat}')
    print(f'p-value: {p}')
    if p < .05:
        print(f'The low p-value of {round(p, 8)} suggests these variances are different.')
    else:
        print(f'The high p-value of {round(p, 8)} suggests these variances are the same.')

#### Bartlett's (for normally distributed measures-- all the bells)
def return_bartlett(subset1, subset2, continuous_feature):

    '''
    This function accepts two subset dataframes and a feature to compare variances across, prints their variances, and prints the statistic and p-value from 
    Bartlett's test and suggests whether the user should reject the null hypothesis that the variances are the same or to fail to reject this null hypothesis.
    '''

    # run the test
    stat, p = stats.bartlett(subset1[continuous_feature], subset2[continuous_feature])

    # print variances
    print(f'Variances:{subset1[continuous_feature].var(), subset2[continuous_feature].var()}')

    # print suggested homoscedastic verdict
    print("Bartlett's Test")
    print("---------------")
    print(f'Statistic: {stat}')
    print(f'p-value: {p}')
    if p < .05:
        print(f'The low p-value of {round(p, 8)} suggests these variances are different.')
    else:
        print(f'The high p-value of {round(p, 8)} suggests these variances are the same.')


# FEATURE SELECTION FUNCTIONS # 
# # # Note: must define X_train and y_train prior to running feature selection functions
# # note: also these output lists are ordered backward

# X_train = predictors or features (same thing if you got the right features)
# y_train = target
# k = number of features you want


def select_kbest(X_train, y_train, k):

    '''
    This function defines 3 parameters, X_train (predictors), y_train (target variable) and k (number of features to spit), and returns a list of the best features my man.
    '''

    from sklearn.feature_selection import SelectKBest, f_regression   # import feature selection tools

    f_select = SelectKBest(f_regression, k = k)    # create the selector
    f_select.fit(X_train, y_train)    # fit the selector
    feat_mask = f_select.get_support()    # create a boolean mask to show if feature was selected
    best_features = X_train.iloc[:,feat_mask].columns.to_list()    # create a list of the best features
    
    return best_features    # gimme gimme

def rfe(X_train, y_train, k):

    '''
    This function defines 3 parameters, X_train (features), y_train (target variable) and k (number of features to bop), and returns a list of the best boppits m8.
    '''

    from sklearn.feature_selection import RFE  # import feature selection tools
    from sklearn.linear_model import LinearRegression  # ditto
    
    lm = LinearRegression()    # crank it
    rfe = RFE(lm, k)    # pop it
    rfe.fit(X_train, y_train)    # bop it
    feat_mask = rfe.support_    # twist it
    best_rfe = X_train.iloc[:,feat_mask].columns.tolist()    # pull it 
    
    return best_rfe    # bop it

