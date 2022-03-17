import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import stats

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr


sns.set()



# Thank you, Adam Gomez
def distributions_plot(df, quant_vars):

    '''
    This function creates a nice sized figure, enumerates the list of features passed into the function, creates a grid of subplots,
    and then charts histograms for features in the list onto the subplots.
    '''

    plt.figure(figsize = (20, 11))   # create figure

    for i, cat in enumerate(quant_vars):    # loop through enumerated list
    
        plot_number = i + 1     # i starts at 0, but plot nos should start at 1
        
        plt.subplot(5, 5, plot_number)  # create subplot
        
        plt.title(cat)  # title
        
        df[cat].hist(color = 'indigo', edgecolor='black')   # display histogram for column

        plt.tight_layout(); # clean
    

def boxplot_grid(df, quant_vars):

    '''
    This function creates a nice sized figure, enumerates the list of features passed into the function, creates a grid of subplots,
    and then charts histograms for features in the list onto the subplots.
    '''
    
    plt.figure(figsize = (28, 16))   # create figure
    
    for i, cat in enumerate(quant_vars):    # loop through enumerated list
    
        plot_number = i + 1     # i starts at 0, but plot nos should start at 1
        
        plt.subplot(5, 5, plot_number)  # create subplot
        
        plt.boxplot(df[cat])   # display boxplot for column
         
        plt.ylabel(cat, size = 18)     # label y-axis

        plt.yticks(size = 16)   # increase size on y-axis ticks

        plt.grid(True)      # show gridlines

        plt.tight_layout(); # clean

# visualize distribution of target variable across different departments
def juxtapose_distributions(C1, C2, C3, target):

    plt.figure(figsize = (8, 5))

    plt.hist([C1[target], C2[target], C3[target]],
            label = ['R&D', 'Sales', 'HR'],
            color = ['red', 'black', 'midnightblue'],
            bins = 2, 
            lw = .5
            )
    plt.legend()
    # plt.title('', size = 16, pad = 6)
    plt.xlabel(target, size = 13)
    plt.ylabel('Frequency', size = 13);


# plot_categorical_and_continuous defines 3 parameters, a dataframe to pull data from, and x variable (categorical column) and a y variable (continuous value column), and returns visualizations of these relationships.
def plot_categorical_and_continuous(df, x, y):

    '''
    This function shows the relationship between two variables (a categorical feature and a continuous one) on 3 different kind of plots (box, strip, violin).
    '''

    # plot 3 figures and 3 different plots for visualizing categorical-continuous relationships
    plt.figure(figsize = (8, 5))
    sns.boxplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.stripplot(x = x, y = y, data = df, palette = 'inferno_r', size = 1.6);
    plt.figure(figsize = (8, 5))
    sns.violinplot(x = x, y = y, data = df, palette = 'inferno_r');





# itertools, where are you? whoops.

# plot_variable_pairs defines two parameters, a dataframe and a list of columns to run through, and returns relational plots with fitted regression lines
def plot_variable_pairs(df, quant_vars):
    
    '''
    This function counts the number of features passed in a list, forms all possible pairs for the number of features it selects,
    and runs a pearson's correlation test on each pair, and then plots the relationship between each pair as well as a regression line, and titles
    each plot with Pearson's R and the respective p-value. This function currently accepts up to 22 features for pairing.
    '''

    # determine k
    k = len(quant_vars)

    # set up if-conditional to see how many features are being paired
    if k == 2:

        # determine correlation coefficient
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])

        # plot relationships between continuous variables
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');


    # pair 3 features
    if k == 3:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])

        # plot relationships between continuous variables
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');


    # pair 4 features
    if k == 4:
        
        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])

        # plot relationships between continuous variables
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');


    # pair 5 features
    if k == 5:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        

        # plot relationships between continuous variables
        
        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');


    # pair 6 features
    if k == 6:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])

        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

    # pair 7 features
    if k == 7:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])


        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------');

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------');

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------');

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');


    # pair 8 features
    if k == 8:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])


        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');
    
    # pair 9 features
    if k == 9:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])


        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

    # pair 10 features
    if k == 10:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])




        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

    # pair 11 features
    if k == 11:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])





        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        # plot XLVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        # plot XLVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        # plot XLVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        # plot XLIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        # plot L
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        # plot LI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        # plot LII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        # plot LIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        # plot LIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        # plot LV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

    if k == 12:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])





        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        # plot XLVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        # plot XLVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        # plot XLVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        # plot XLIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        # plot L
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        # plot LI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        # plot LII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        # plot LIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        # plot LIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        # plot LV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        # plot LVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        # plot LVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        # plot LVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        # plot LIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        # plot LX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        # plot LXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        # plot LXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        # plot LXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        # plot LXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        # plot LXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        # plot LXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');


    if k == 13:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])




        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        # plot XLVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        # plot XLVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        # plot XLVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        # plot XLIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        # plot L
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        # plot LI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        # plot LII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        # plot LIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        # plot LIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        # plot LV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        # plot LVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        # plot LVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        # plot LVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        # plot LIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        # plot LX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        # plot LXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        # plot LXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        # plot LXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        # plot LXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        # plot LXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        # plot LXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        # plot LXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        # plot LXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        # plot LXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        # plot LXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        # plot LXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        # plot LXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        # plot LXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        # plot LXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        # plot LXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        # plot LXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        # plot LXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        # plot LXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

    if k == 14:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])
        corr79, p79 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[0]])
        corr80, p80 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[1]])
        corr81, p81 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[2]])
        corr82, p82 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[3]])
        corr83, p83 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[4]])
        corr84, p84 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[5]])
        corr85, p85 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[6]])
        corr86, p86 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[7]])
        corr87, p87 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[8]])
        corr88, p88 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[9]])
        corr89, p89 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[10]])
        corr90, p90 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[11]])
        corr91, p91 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[12]])



        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        # plot XLVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        # plot XLVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        # plot XLVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        # plot XLIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        # plot L
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        # plot LI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        # plot LII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        # plot LIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        # plot LIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        # plot LV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        # plot LVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        # plot LVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        # plot LVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        # plot LIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        # plot LX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        # plot LXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        # plot LXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        # plot LXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        # plot LXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        # plot LXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        # plot LXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        # plot LXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        # plot LXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        # plot LXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        # plot LXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        # plot LXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        # plot LXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        # plot LXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        # plot LXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        # plot LXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        # plot LXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        # plot LXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        # plot LXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

        # plot LXXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr79, 3)} | P-value: {round(p79, 4)} \n -----------------');

        # plot LXXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr80, 3)} | P-value: {round(p80, 4)} \n -----------------');

        # plot LXXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr81, 3)} | P-value: {round(p81, 4)} \n -----------------');

        # plot LXXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr82, 3)} | P-value: {round(p82, 4)} \n -----------------');

        # plot LXXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr83, 3)} | P-value: {round(p83, 4)} \n -----------------');

        # plot LXXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr84, 3)} | P-value: {round(p84, 4)} \n -----------------');

        # plot LXXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr85, 3)} | P-value: {round(p85, 4)} \n -----------------');

        # plot LXXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr86, 3)} | P-value: {round(p86, 4)} \n -----------------');

        # plot LXXXVII
        sns.lmplot(x = quant_vars[8], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr87, 3)} | P-value: {round(p87, 4)} \n -----------------');

        # plot LXXXVIII
        sns.lmplot(x = quant_vars[9], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr88, 3)} | P-value: {round(p88, 4)} \n -----------------');

        # plot LXXXIX
        sns.lmplot(x = quant_vars[10], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr89, 3)} | P-value: {round(p89, 4)} \n -----------------');

        # plot XC
        sns.lmplot(x = quant_vars[11], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr90, 3)} | P-value: {round(p90, 4)} \n -----------------');

        # plot XCI
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr91, 3)} | P-value: {round(p91, 4)} \n -----------------');

    if k == 15:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])
        corr79, p79 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[0]])
        corr80, p80 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[1]])
        corr81, p81 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[2]])
        corr82, p82 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[3]])
        corr83, p83 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[4]])
        corr84, p84 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[5]])
        corr85, p85 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[6]])
        corr86, p86 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[7]])
        corr87, p87 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[8]])
        corr88, p88 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[9]])
        corr89, p89 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[10]])
        corr90, p90 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[11]])
        corr91, p91 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[12]])
        corr92, p92 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[0]])
        corr93, p93 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[1]])
        corr94, p94 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[2]])
        corr95, p95 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[3]])
        corr96, p96 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[4]])
        corr97, p97 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[5]])
        corr98, p98 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[6]])
        corr99, p99 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[7]])
        corr100, p100 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[8]])
        corr101, p101 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[9]])
        corr102, p102 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[10]])
        corr103, p103 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[11]])
        corr104, p104 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[12]])
        corr105, p105 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[13]])


        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        # plot XLVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        # plot XLVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        # plot XLVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        # plot XLIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        # plot L
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        # plot LI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        # plot LII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        # plot LIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        # plot LIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        # plot LV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        # plot LVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        # plot LVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        # plot LVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        # plot LIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        # plot LX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        # plot LXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        # plot LXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        # plot LXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        # plot LXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        # plot LXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        # plot LXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        # plot LXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        # plot LXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        # plot LXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        # plot LXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        # plot LXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        # plot LXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        # plot LXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        # plot LXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        # plot LXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        # plot LXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        # plot LXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        # plot LXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

        # plot LXXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr79, 3)} | P-value: {round(p79, 4)} \n -----------------');

        # plot LXXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr80, 3)} | P-value: {round(p80, 4)} \n -----------------');

        # plot LXXXI
        sns.lmplot(x = quant_vars[3], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr81, 3)} | P-value: {round(p81, 4)} \n -----------------');

        # plot LXXXII
        sns.lmplot(x = quant_vars[4], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr82, 3)} | P-value: {round(p82, 4)} \n -----------------');

        # plot LXXXIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr83, 3)} | P-value: {round(p83, 4)} \n -----------------');

        # plot LXXXIV
        sns.lmplot(x = quant_vars[6], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr84, 3)} | P-value: {round(p84, 4)} \n -----------------');

        # plot LXXXV
        sns.lmplot(x = quant_vars[7], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr85, 3)} | P-value: {round(p85, 4)} \n -----------------');

        # plot LXXXVI
        sns.lmplot(x = quant_vars[8], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr86, 3)} | P-value: {round(p86, 4)} \n -----------------');

        # plot LXXXVII
        sns.lmplot(x = quant_vars[9], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr87, 3)} | P-value: {round(p87, 4)} \n -----------------');

        # plot LXXXVIII
        sns.lmplot(x = quant_vars[10], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr88, 3)} | P-value: {round(p88, 4)} \n -----------------');

        # plot LXXXIX
        sns.lmplot(x = quant_vars[11], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr89, 3)} | P-value: {round(p89, 4)} \n -----------------');

        # plot XC
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr90, 3)} | P-value: {round(p90, 4)} \n -----------------');

        # plot XCI
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr91, 3)} | P-value: {round(p91, 4)} \n -----------------');

        # plot XCII
        sns.lmplot(x = quant_vars[0], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr92, 3)} | P-value: {round(p92, 4)} \n -----------------');

        # plot XCIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr93, 3)} | P-value: {round(p93, 4)} \n -----------------');

        # plot XCIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr94, 3)} | P-value: {round(p94, 4)} \n -----------------');

        # plot XCV
        sns.lmplot(x = quant_vars[3], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr95, 3)} | P-value: {round(p95, 4)} \n -----------------');

        # plot XCVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr96, 3)} | P-value: {round(p96, 4)} \n -----------------');

        # plot XCVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr97, 3)} | P-value: {round(p97, 4)} \n -----------------');

        # plot XCVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr98, 3)} | P-value: {round(p98, 4)} \n -----------------');

        # plot XCIX
        sns.lmplot(x = quant_vars[7], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr99, 3)} | P-value: {round(p99, 4)} \n -----------------');

        # plot C
        sns.lmplot(x = quant_vars[8], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr100, 3)} | P-value: {round(p100, 4)} \n -----------------');

        # plot CI
        sns.lmplot(x = quant_vars[9], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr101, 3)} | P-value: {round(p101, 4)} \n -----------------');

        # plot CII
        sns.lmplot(x = quant_vars[10], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr102, 3)} | P-value: {round(p102, 4)} \n -----------------');

        # plot CIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr103, 3)} | P-value: {round(p103, 4)} \n -----------------');

        # plot CIV
        sns.lmplot(x = quant_vars[12], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr104, 3)} | P-value: {round(p104, 4)} \n -----------------');

        # plot CV
        sns.lmplot(x = quant_vars[13], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr105, 3)} | P-value: {round(p105, 4)} \n -----------------');


    if k == 16:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])
        corr79, p79 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[0]])
        corr80, p80 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[1]])
        corr81, p81 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[2]])
        corr82, p82 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[3]])
        corr83, p83 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[4]])
        corr84, p84 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[5]])
        corr85, p85 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[6]])
        corr86, p86 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[7]])
        corr87, p87 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[8]])
        corr88, p88 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[9]])
        corr89, p89 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[10]])
        corr90, p90 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[11]])
        corr91, p91 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[12]])
        corr92, p92 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[0]])
        corr93, p93 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[1]])
        corr94, p94 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[2]])
        corr95, p95 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[3]])
        corr96, p96 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[4]])
        corr97, p97 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[5]])
        corr98, p98 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[6]])
        corr99, p99 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[7]])
        corr100, p100 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[8]])
        corr101, p101 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[9]])
        corr102, p102 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[10]])
        corr103, p103 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[11]])
        corr104, p104 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[12]])
        corr105, p105 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[13]])
        corr106, p106 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[0]])
        corr107, p107 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[1]])
        corr108, p108 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[2]])
        corr109, p109 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[3]])
        corr110, p110 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[4]])
        corr111, p111 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[5]])
        corr112, p112 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[6]])
        corr113, p113 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[7]])
        corr114, p114 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[8]])
        corr115, p115 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[9]])
        corr116, p116 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[10]])
        corr117, p117 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[11]])
        corr118, p118 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[12]])
        corr119, p119 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[13]])
        corr120, p120 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[14]])

        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        # plot XLVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        # plot XLVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        # plot XLVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        # plot XLIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        # plot L
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        # plot LI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        # plot LII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        # plot LIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        # plot LIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        # plot LV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        # plot LVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        # plot LVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        # plot LVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        # plot LIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        # plot LX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        # plot LXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        # plot LXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        # plot LXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        # plot LXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        # plot LXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        # plot LXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        # plot LXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        # plot LXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        # plot LXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        # plot LXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        # plot LXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        # plot LXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        # plot LXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        # plot LXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        # plot LXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        # plot LXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        # plot LXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        # plot LXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

        # plot LXXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr79, 3)} | P-value: {round(p79, 4)} \n -----------------');

        # plot LXXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr80, 3)} | P-value: {round(p80, 4)} \n -----------------');

        # plot LXXXI
        sns.lmplot(x = quant_vars[3], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr81, 3)} | P-value: {round(p81, 4)} \n -----------------');

        # plot LXXXII
        sns.lmplot(x = quant_vars[4], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr82, 3)} | P-value: {round(p82, 4)} \n -----------------');

        # plot LXXXIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr83, 3)} | P-value: {round(p83, 4)} \n -----------------');

        # plot LXXXIV
        sns.lmplot(x = quant_vars[6], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr84, 3)} | P-value: {round(p84, 4)} \n -----------------');

        # plot LXXXV
        sns.lmplot(x = quant_vars[7], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr85, 3)} | P-value: {round(p85, 4)} \n -----------------');

        # plot LXXXVI
        sns.lmplot(x = quant_vars[8], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr86, 3)} | P-value: {round(p86, 4)} \n -----------------');

        # plot LXXXVII
        sns.lmplot(x = quant_vars[9], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr87, 3)} | P-value: {round(p87, 4)} \n -----------------');

        # plot LXXXVIII
        sns.lmplot(x = quant_vars[10], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr88, 3)} | P-value: {round(p88, 4)} \n -----------------');

        # plot LXXXIX
        sns.lmplot(x = quant_vars[11], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr89, 3)} | P-value: {round(p89, 4)} \n -----------------');

        # plot XC
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr90, 3)} | P-value: {round(p90, 4)} \n -----------------');

        # plot XCI
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr91, 3)} | P-value: {round(p91, 4)} \n -----------------');

        # plot XCII
        sns.lmplot(x = quant_vars[0], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr92, 3)} | P-value: {round(p92, 4)} \n -----------------');

        # plot XCIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr93, 3)} | P-value: {round(p93, 4)} \n -----------------');

        # plot XCIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr94, 3)} | P-value: {round(p94, 4)} \n -----------------');

        # plot XCV
        sns.lmplot(x = quant_vars[3], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr95, 3)} | P-value: {round(p95, 4)} \n -----------------');

        # plot XCVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr96, 3)} | P-value: {round(p96, 4)} \n -----------------');

        # plot XCVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr97, 3)} | P-value: {round(p97, 4)} \n -----------------');

        # plot XCVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr98, 3)} | P-value: {round(p98, 4)} \n -----------------');

        # plot XCIX
        sns.lmplot(x = quant_vars[7], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr99, 3)} | P-value: {round(p99, 4)} \n -----------------');

        # plot C
        sns.lmplot(x = quant_vars[8], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr100, 3)} | P-value: {round(p100, 4)} \n -----------------');

        # plot CI
        sns.lmplot(x = quant_vars[9], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr101, 3)} | P-value: {round(p101, 4)} \n -----------------');

        # plot CII
        sns.lmplot(x = quant_vars[10], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr102, 3)} | P-value: {round(p102, 4)} \n -----------------');

        # plot CIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr103, 3)} | P-value: {round(p103, 4)} \n -----------------');

        # plot CIV
        sns.lmplot(x = quant_vars[12], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr104, 3)} | P-value: {round(p104, 4)} \n -----------------');

        # plot CV
        sns.lmplot(x = quant_vars[13], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr105, 3)} | P-value: {round(p105, 4)} \n -----------------');

        # plot CVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr106, 3)} | P-value: {round(p106, 4)} \n -----------------');

        # plot CVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr107, 3)} | P-value: {round(p107, 4)} \n -----------------');

        # plot CVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr108, 3)} | P-value: {round(p108, 4)} \n -----------------');

        # plot CVIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr109, 3)} | P-value: {round(p109, 4)} \n -----------------');

        # plot CX
        sns.lmplot(x = quant_vars[4], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr110, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr111, 3)} | P-value: {round(p111, 4)} \n -----------------');

        # plot CXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr112, 3)} | P-value: {round(p112, 4)} \n -----------------');

        # plot CXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr113, 3)} | P-value: {round(p113, 4)} \n -----------------');

        # plot CXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr114, 3)} | P-value: {round(p114, 4)} \n -----------------');

        # plot CXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr115, 3)} | P-value: {round(p115, 4)} \n -----------------');

        # plot CXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr116, 3)} | P-value: {round(p116, 4)} \n -----------------');

        # plot CXVII
        sns.lmplot(x = quant_vars[11], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr117, 3)} | P-value: {round(p117, 4)} \n -----------------');

        # plot CXVIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr118, 3)} | P-value: {round(p118, 4)} \n -----------------');

        # plot CXIX
        sns.lmplot(x = quant_vars[13], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr119, 3)} | P-value: {round(p119, 4)} \n -----------------');

        # plot CXX
        sns.lmplot(x = quant_vars[14], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr120, 3)} | P-value: {round(p110, 4)} \n -----------------');


    if k == 17:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])
        corr79, p79 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[0]])
        corr80, p80 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[1]])
        corr81, p81 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[2]])
        corr82, p82 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[3]])
        corr83, p83 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[4]])
        corr84, p84 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[5]])
        corr85, p85 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[6]])
        corr86, p86 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[7]])
        corr87, p87 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[8]])
        corr88, p88 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[9]])
        corr89, p89 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[10]])
        corr90, p90 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[11]])
        corr91, p91 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[12]])
        corr92, p92 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[0]])
        corr93, p93 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[1]])
        corr94, p94 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[2]])
        corr95, p95 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[3]])
        corr96, p96 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[4]])
        corr97, p97 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[5]])
        corr98, p98 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[6]])
        corr99, p99 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[7]])
        corr100, p100 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[8]])
        corr101, p101 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[9]])
        corr102, p102 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[10]])
        corr103, p103 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[11]])
        corr104, p104 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[12]])
        corr105, p105 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[13]])
        corr106, p106 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[0]])
        corr107, p107 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[1]])
        corr108, p108 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[2]])
        corr109, p109 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[3]])
        corr110, p110 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[4]])
        corr111, p111 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[5]])
        corr112, p112 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[6]])
        corr113, p113 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[7]])
        corr114, p114 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[8]])
        corr115, p115 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[9]])
        corr116, p116 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[10]])
        corr117, p117 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[11]])
        corr118, p118 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[12]])
        corr119, p119 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[13]])
        corr120, p120 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[14]])
        corr121, p121 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[0]])
        corr122, p122 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[1]])
        corr123, p123 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[2]])
        corr124, p124 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[3]])
        corr125, p125 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[4]])
        corr126, p126 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[5]])
        corr127, p127 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[6]])
        corr128, p128 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[7]])
        corr129, p129 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[8]])
        corr130, p130 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[9]])
        corr131, p131 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[10]])
        corr132, p132 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[11]])
        corr133, p133 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[12]])
        corr134, p134 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[13]])
        corr135, p135 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[14]])
        corr136, p136 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[15]])

        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        # plot XLVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        # plot XLVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        # plot XLVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        # plot XLIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        # plot L
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        # plot LI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        # plot LII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        # plot LIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        # plot LIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        # plot LV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        # plot LVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        # plot LVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        # plot LVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        # plot LIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        # plot LX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        # plot LXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        # plot LXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        # plot LXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        # plot LXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        # plot LXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        # plot LXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        # plot LXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        # plot LXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        # plot LXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        # plot LXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        # plot LXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        # plot LXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        # plot LXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        # plot LXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        # plot LXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        # plot LXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        # plot LXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        # plot LXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

        # plot LXXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr79, 3)} | P-value: {round(p79, 4)} \n -----------------');

        # plot LXXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr80, 3)} | P-value: {round(p80, 4)} \n -----------------');

        # plot LXXXI
        sns.lmplot(x = quant_vars[3], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr81, 3)} | P-value: {round(p81, 4)} \n -----------------');

        # plot LXXXII
        sns.lmplot(x = quant_vars[4], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr82, 3)} | P-value: {round(p82, 4)} \n -----------------');

        # plot LXXXIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr83, 3)} | P-value: {round(p83, 4)} \n -----------------');

        # plot LXXXIV
        sns.lmplot(x = quant_vars[6], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr84, 3)} | P-value: {round(p84, 4)} \n -----------------');

        # plot LXXXV
        sns.lmplot(x = quant_vars[7], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr85, 3)} | P-value: {round(p85, 4)} \n -----------------');

        # plot LXXXVI
        sns.lmplot(x = quant_vars[8], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr86, 3)} | P-value: {round(p86, 4)} \n -----------------');

        # plot LXXXVII
        sns.lmplot(x = quant_vars[9], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr87, 3)} | P-value: {round(p87, 4)} \n -----------------');

        # plot LXXXVIII
        sns.lmplot(x = quant_vars[10], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr88, 3)} | P-value: {round(p88, 4)} \n -----------------');

        # plot LXXXIX
        sns.lmplot(x = quant_vars[11], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr89, 3)} | P-value: {round(p89, 4)} \n -----------------');

        # plot XC
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr90, 3)} | P-value: {round(p90, 4)} \n -----------------');

        # plot XCI
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr91, 3)} | P-value: {round(p91, 4)} \n -----------------');

        # plot XCII
        sns.lmplot(x = quant_vars[0], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr92, 3)} | P-value: {round(p92, 4)} \n -----------------');

        # plot XCIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr93, 3)} | P-value: {round(p93, 4)} \n -----------------');

        # plot XCIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr94, 3)} | P-value: {round(p94, 4)} \n -----------------');

        # plot XCV
        sns.lmplot(x = quant_vars[3], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr95, 3)} | P-value: {round(p95, 4)} \n -----------------');

        # plot XCVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr96, 3)} | P-value: {round(p96, 4)} \n -----------------');

        # plot XCVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr97, 3)} | P-value: {round(p97, 4)} \n -----------------');

        # plot XCVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr98, 3)} | P-value: {round(p98, 4)} \n -----------------');

        # plot XCIX
        sns.lmplot(x = quant_vars[7], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr99, 3)} | P-value: {round(p99, 4)} \n -----------------');

        # plot C
        sns.lmplot(x = quant_vars[8], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr100, 3)} | P-value: {round(p100, 4)} \n -----------------');

        # plot CI
        sns.lmplot(x = quant_vars[9], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr101, 3)} | P-value: {round(p101, 4)} \n -----------------');

        # plot CII
        sns.lmplot(x = quant_vars[10], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr102, 3)} | P-value: {round(p102, 4)} \n -----------------');

        # plot CIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr103, 3)} | P-value: {round(p103, 4)} \n -----------------');

        # plot CIV
        sns.lmplot(x = quant_vars[12], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr104, 3)} | P-value: {round(p104, 4)} \n -----------------');

        # plot CV
        sns.lmplot(x = quant_vars[13], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr105, 3)} | P-value: {round(p105, 4)} \n -----------------');

        # plot CVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr106, 3)} | P-value: {round(p106, 4)} \n -----------------');

        # plot CVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr107, 3)} | P-value: {round(p107, 4)} \n -----------------');

        # plot CVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr108, 3)} | P-value: {round(p108, 4)} \n -----------------');

        # plot CVIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr109, 3)} | P-value: {round(p109, 4)} \n -----------------');

        # plot CX
        sns.lmplot(x = quant_vars[4], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr110, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr111, 3)} | P-value: {round(p111, 4)} \n -----------------');

        # plot CXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr112, 3)} | P-value: {round(p112, 4)} \n -----------------');

        # plot CXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr113, 3)} | P-value: {round(p113, 4)} \n -----------------');

        # plot CXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr114, 3)} | P-value: {round(p114, 4)} \n -----------------');

        # plot CXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr115, 3)} | P-value: {round(p115, 4)} \n -----------------');

        # plot CXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr116, 3)} | P-value: {round(p116, 4)} \n -----------------');

        # plot CXVII
        sns.lmplot(x = quant_vars[11], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr117, 3)} | P-value: {round(p117, 4)} \n -----------------');

        # plot CXVIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr118, 3)} | P-value: {round(p118, 4)} \n -----------------');

        # plot CXIX
        sns.lmplot(x = quant_vars[13], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr119, 3)} | P-value: {round(p119, 4)} \n -----------------');

        # plot CXX
        sns.lmplot(x = quant_vars[14], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr120, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXXI
        sns.lmplot(x = quant_vars[0], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr121, 3)} | P-value: {round(p121, 4)} \n -----------------');

        # plot CXXII
        sns.lmplot(x = quant_vars[1], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr122, 3)} | P-value: {round(p122, 4)} \n -----------------');

        # plot CXXIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr123, 3)} | P-value: {round(p123, 4)} \n -----------------');

        # plot CXXIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr124, 3)} | P-value: {round(p124, 4)} \n -----------------');

        # plot CXXV
        sns.lmplot(x = quant_vars[4], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr125, 3)} | P-value: {round(p125, 4)} \n -----------------');

        # plot CXXVI
        sns.lmplot(x = quant_vars[5], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr126, 3)} | P-value: {round(p126, 4)} \n -----------------');

        # plot CXXVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr127, 3)} | P-value: {round(p127, 4)} \n -----------------');

        # plot CXXVIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr128, 3)} | P-value: {round(p128, 4)} \n -----------------');

        # plot CXXIX
        sns.lmplot(x = quant_vars[8], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr129, 3)} | P-value: {round(p129, 4)} \n -----------------');

        # plot CXXX
        sns.lmplot(x = quant_vars[9], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr130, 3)} | P-value: {round(p130, 4)} \n -----------------');

        # plot CXXXI
        sns.lmplot(x = quant_vars[10], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr131, 3)} | P-value: {round(p131, 4)} \n -----------------');

        # plot CXXXII
        sns.lmplot(x = quant_vars[11], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr132, 3)} | P-value: {round(p132, 4)} \n -----------------');

        # plot CXXXIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr133, 3)} | P-value: {round(p133, 4)} \n -----------------');

        # plot CXXXIV
        sns.lmplot(x = quant_vars[13], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr134, 3)} | P-value: {round(p134, 4)} \n -----------------');

        # plot CXXXV
        sns.lmplot(x = quant_vars[14], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr135, 3)} | P-value: {round(p135, 4)} \n -----------------');

        # plot CXXXVI
        sns.lmplot(x = quant_vars[15], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr136, 3)} | P-value: {round(p136, 4)} \n -----------------');

    if k == 18:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])
        corr79, p79 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[0]])
        corr80, p80 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[1]])
        corr81, p81 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[2]])
        corr82, p82 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[3]])
        corr83, p83 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[4]])
        corr84, p84 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[5]])
        corr85, p85 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[6]])
        corr86, p86 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[7]])
        corr87, p87 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[8]])
        corr88, p88 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[9]])
        corr89, p89 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[10]])
        corr90, p90 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[11]])
        corr91, p91 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[12]])
        corr92, p92 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[0]])
        corr93, p93 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[1]])
        corr94, p94 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[2]])
        corr95, p95 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[3]])
        corr96, p96 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[4]])
        corr97, p97 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[5]])
        corr98, p98 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[6]])
        corr99, p99 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[7]])
        corr100, p100 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[8]])
        corr101, p101 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[9]])
        corr102, p102 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[10]])
        corr103, p103 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[11]])
        corr104, p104 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[12]])
        corr105, p105 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[13]])
        corr106, p106 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[0]])
        corr107, p107 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[1]])
        corr108, p108 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[2]])
        corr109, p109 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[3]])
        corr110, p110 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[4]])
        corr111, p111 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[5]])
        corr112, p112 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[6]])
        corr113, p113 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[7]])
        corr114, p114 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[8]])
        corr115, p115 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[9]])
        corr116, p116 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[10]])
        corr117, p117 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[11]])
        corr118, p118 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[12]])
        corr119, p119 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[13]])
        corr120, p120 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[14]])
        corr121, p121 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[0]])
        corr122, p122 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[1]])
        corr123, p123 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[2]])
        corr124, p124 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[3]])
        corr125, p125 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[4]])
        corr126, p126 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[5]])
        corr127, p127 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[6]])
        corr128, p128 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[7]])
        corr129, p129 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[8]])
        corr130, p130 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[9]])
        corr131, p131 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[10]])
        corr132, p132 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[11]])
        corr133, p133 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[12]])
        corr134, p134 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[13]])
        corr135, p135 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[14]])
        corr136, p136 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[15]])
        corr137, p137 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[0]])
        corr138, p138 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[1]])
        corr139, p139 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[2]])
        corr140, p140 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[3]])
        corr141, p141 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[4]])
        corr142, p142 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[5]])
        corr143, p143 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[6]])
        corr144, p144 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[7]])
        corr145, p145 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[8]])
        corr146, p146 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[9]])
        corr147, p147 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[10]])
        corr148, p148 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[11]])
        corr149, p149 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[12]])
        corr150, p150 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[13]])
        corr151, p151 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[14]])
        corr152, p152 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[15]])
        corr153, p153 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[16]])
        
        # plot relationships between continuous variables
        
        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        # plot XLVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        # plot XLVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        # plot XLVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        # plot XLIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        # plot L
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        # plot LI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        # plot LII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        # plot LIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        # plot LIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        # plot LV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        # plot LVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        # plot LVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        # plot LVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        # plot LIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        # plot LX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        # plot LXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        # plot LXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        # plot LXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        # plot LXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        # plot LXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        # plot LXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        # plot LXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        # plot LXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        # plot LXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        # plot LXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        # plot LXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        # plot LXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        # plot LXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        # plot LXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        # plot LXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        # plot LXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        # plot LXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        # plot LXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

        # plot LXXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr79, 3)} | P-value: {round(p79, 4)} \n -----------------');

        # plot LXXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr80, 3)} | P-value: {round(p80, 4)} \n -----------------');

        # plot LXXXI
        sns.lmplot(x = quant_vars[3], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr81, 3)} | P-value: {round(p81, 4)} \n -----------------');

        # plot LXXXII
        sns.lmplot(x = quant_vars[4], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr82, 3)} | P-value: {round(p82, 4)} \n -----------------');

        # plot LXXXIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr83, 3)} | P-value: {round(p83, 4)} \n -----------------');

        # plot LXXXIV
        sns.lmplot(x = quant_vars[6], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr84, 3)} | P-value: {round(p84, 4)} \n -----------------');

        # plot LXXXV
        sns.lmplot(x = quant_vars[7], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr85, 3)} | P-value: {round(p85, 4)} \n -----------------');

        # plot LXXXVI
        sns.lmplot(x = quant_vars[8], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr86, 3)} | P-value: {round(p86, 4)} \n -----------------');

        # plot LXXXVII
        sns.lmplot(x = quant_vars[9], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr87, 3)} | P-value: {round(p87, 4)} \n -----------------');

        # plot LXXXVIII
        sns.lmplot(x = quant_vars[10], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr88, 3)} | P-value: {round(p88, 4)} \n -----------------');

        # plot LXXXIX
        sns.lmplot(x = quant_vars[11], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr89, 3)} | P-value: {round(p89, 4)} \n -----------------');

        # plot XC
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr90, 3)} | P-value: {round(p90, 4)} \n -----------------');

        # plot XCI
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr91, 3)} | P-value: {round(p91, 4)} \n -----------------');

        # plot XCII
        sns.lmplot(x = quant_vars[0], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr92, 3)} | P-value: {round(p92, 4)} \n -----------------');

        # plot XCIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr93, 3)} | P-value: {round(p93, 4)} \n -----------------');

        # plot XCIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr94, 3)} | P-value: {round(p94, 4)} \n -----------------');

        # plot XCV
        sns.lmplot(x = quant_vars[3], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr95, 3)} | P-value: {round(p95, 4)} \n -----------------');

        # plot XCVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr96, 3)} | P-value: {round(p96, 4)} \n -----------------');

        # plot XCVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr97, 3)} | P-value: {round(p97, 4)} \n -----------------');

        # plot XCVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr98, 3)} | P-value: {round(p98, 4)} \n -----------------');

        # plot XCIX
        sns.lmplot(x = quant_vars[7], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr99, 3)} | P-value: {round(p99, 4)} \n -----------------');

        # plot C
        sns.lmplot(x = quant_vars[8], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr100, 3)} | P-value: {round(p100, 4)} \n -----------------');

        # plot CI
        sns.lmplot(x = quant_vars[9], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr101, 3)} | P-value: {round(p101, 4)} \n -----------------');

        # plot CII
        sns.lmplot(x = quant_vars[10], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr102, 3)} | P-value: {round(p102, 4)} \n -----------------');

        # plot CIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr103, 3)} | P-value: {round(p103, 4)} \n -----------------');

        # plot CIV
        sns.lmplot(x = quant_vars[12], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr104, 3)} | P-value: {round(p104, 4)} \n -----------------');

        # plot CV
        sns.lmplot(x = quant_vars[13], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr105, 3)} | P-value: {round(p105, 4)} \n -----------------');

        # plot CVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr106, 3)} | P-value: {round(p106, 4)} \n -----------------');

        # plot CVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr107, 3)} | P-value: {round(p107, 4)} \n -----------------');

        # plot CVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr108, 3)} | P-value: {round(p108, 4)} \n -----------------');

        # plot CVIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr109, 3)} | P-value: {round(p109, 4)} \n -----------------');

        # plot CX
        sns.lmplot(x = quant_vars[4], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr110, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr111, 3)} | P-value: {round(p111, 4)} \n -----------------');

        # plot CXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr112, 3)} | P-value: {round(p112, 4)} \n -----------------');

        # plot CXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr113, 3)} | P-value: {round(p113, 4)} \n -----------------');

        # plot CXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr114, 3)} | P-value: {round(p114, 4)} \n -----------------');

        # plot CXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr115, 3)} | P-value: {round(p115, 4)} \n -----------------');

        # plot CXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr116, 3)} | P-value: {round(p116, 4)} \n -----------------');

        # plot CXVII
        sns.lmplot(x = quant_vars[11], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr117, 3)} | P-value: {round(p117, 4)} \n -----------------');

        # plot CXVIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr118, 3)} | P-value: {round(p118, 4)} \n -----------------');

        # plot CXIX
        sns.lmplot(x = quant_vars[13], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr119, 3)} | P-value: {round(p119, 4)} \n -----------------');

        # plot CXX
        sns.lmplot(x = quant_vars[14], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr120, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXXI
        sns.lmplot(x = quant_vars[0], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr121, 3)} | P-value: {round(p121, 4)} \n -----------------');

        # plot CXXII
        sns.lmplot(x = quant_vars[1], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr122, 3)} | P-value: {round(p122, 4)} \n -----------------');

        # plot CXXIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr123, 3)} | P-value: {round(p123, 4)} \n -----------------');

        # plot CXXIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr124, 3)} | P-value: {round(p124, 4)} \n -----------------');

        # plot CXXV
        sns.lmplot(x = quant_vars[4], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr125, 3)} | P-value: {round(p125, 4)} \n -----------------');

        # plot CXXVI
        sns.lmplot(x = quant_vars[5], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr126, 3)} | P-value: {round(p126, 4)} \n -----------------');

        # plot CXXVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr127, 3)} | P-value: {round(p127, 4)} \n -----------------');

        # plot CXXVIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr128, 3)} | P-value: {round(p128, 4)} \n -----------------');

        # plot CXXIX
        sns.lmplot(x = quant_vars[8], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr129, 3)} | P-value: {round(p129, 4)} \n -----------------');

        # plot CXXX
        sns.lmplot(x = quant_vars[9], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr130, 3)} | P-value: {round(p130, 4)} \n -----------------');

        # plot CXXXI
        sns.lmplot(x = quant_vars[10], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr131, 3)} | P-value: {round(p131, 4)} \n -----------------');

        # plot CXXXII
        sns.lmplot(x = quant_vars[11], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr132, 3)} | P-value: {round(p132, 4)} \n -----------------');

        # plot CXXXIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr133, 3)} | P-value: {round(p133, 4)} \n -----------------');

        # plot CXXXIV
        sns.lmplot(x = quant_vars[13], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr134, 3)} | P-value: {round(p134, 4)} \n -----------------');

        # plot CXXXV
        sns.lmplot(x = quant_vars[14], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr135, 3)} | P-value: {round(p135, 4)} \n -----------------');

        # plot CXXXVI
        sns.lmplot(x = quant_vars[15], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr136, 3)} | P-value: {round(p136, 4)} \n -----------------');

        # plot CXXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr137, 3)} | P-value: {round(p137, 4)} \n -----------------');

        # plot CXXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr138, 3)} | P-value: {round(p138, 4)} \n -----------------');

        # plot CXXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr139, 3)} | P-value: {round(p139, 4)} \n -----------------');

        # plot CXL
        sns.lmplot(x = quant_vars[3], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr140, 3)} | P-value: {round(p140, 4)} \n -----------------');

        # plot CXLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr141, 3)} | P-value: {round(p141, 4)} \n -----------------');

        # plot CXLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr142, 3)} | P-value: {round(p142, 4)} \n -----------------');

        # plot CXLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr143, 3)} | P-value: {round(p143, 4)} \n -----------------');

        # plot CXLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr144, 3)} | P-value: {round(p144, 4)} \n -----------------');

        # plot CXLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr145, 3)} | P-value: {round(p145, 4)} \n -----------------');

        # plot CXLVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr146, 3)} | P-value: {round(p146, 4)} \n -----------------');

        # plot CXLVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr147, 3)} | P-value: {round(p147, 4)} \n -----------------');

        # plot CXLVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr148, 3)} | P-value: {round(p148, 4)} \n -----------------');

        # plot CXLIX
        sns.lmplot(x = quant_vars[12], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr149, 3)} | P-value: {round(p149, 4)} \n -----------------');

        # plot CL
        sns.lmplot(x = quant_vars[13], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr150, 3)} | P-value: {round(p150, 4)} \n -----------------');

        # plot CLI
        sns.lmplot(x = quant_vars[14], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr151, 3)} | P-value: {round(p151, 4)} \n -----------------');

        # plot CLII
        sns.lmplot(x = quant_vars[15], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr152, 3)} | P-value: {round(p152, 4)} \n -----------------');

        # plot CLIII
        sns.lmplot(x = quant_vars[16], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr153, 3)} | P-value: {round(p153, 4)} \n -----------------');


    if k == 19:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])
        corr79, p79 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[0]])
        corr80, p80 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[1]])
        corr81, p81 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[2]])
        corr82, p82 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[3]])
        corr83, p83 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[4]])
        corr84, p84 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[5]])
        corr85, p85 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[6]])
        corr86, p86 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[7]])
        corr87, p87 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[8]])
        corr88, p88 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[9]])
        corr89, p89 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[10]])
        corr90, p90 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[11]])
        corr91, p91 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[12]])
        corr92, p92 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[0]])
        corr93, p93 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[1]])
        corr94, p94 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[2]])
        corr95, p95 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[3]])
        corr96, p96 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[4]])
        corr97, p97 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[5]])
        corr98, p98 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[6]])
        corr99, p99 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[7]])
        corr100, p100 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[8]])
        corr101, p101 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[9]])
        corr102, p102 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[10]])
        corr103, p103 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[11]])
        corr104, p104 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[12]])
        corr105, p105 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[13]])
        corr106, p106 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[0]])
        corr107, p107 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[1]])
        corr108, p108 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[2]])
        corr109, p109 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[3]])
        corr110, p110 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[4]])
        corr111, p111 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[5]])
        corr112, p112 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[6]])
        corr113, p113 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[7]])
        corr114, p114 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[8]])
        corr115, p115 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[9]])
        corr116, p116 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[10]])
        corr117, p117 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[11]])
        corr118, p118 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[12]])
        corr119, p119 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[13]])
        corr120, p120 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[14]])
        corr121, p121 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[0]])
        corr122, p122 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[1]])
        corr123, p123 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[2]])
        corr124, p124 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[3]])
        corr125, p125 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[4]])
        corr126, p126 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[5]])
        corr127, p127 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[6]])
        corr128, p128 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[7]])
        corr129, p129 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[8]])
        corr130, p130 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[9]])
        corr131, p131 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[10]])
        corr132, p132 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[11]])
        corr133, p133 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[12]])
        corr134, p134 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[13]])
        corr135, p135 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[14]])
        corr136, p136 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[15]])
        corr137, p137 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[0]])
        corr138, p138 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[1]])
        corr139, p139 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[2]])
        corr140, p140 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[3]])
        corr141, p141 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[4]])
        corr142, p142 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[5]])
        corr143, p143 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[6]])
        corr144, p144 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[7]])
        corr145, p145 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[8]])
        corr146, p146 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[9]])
        corr147, p147 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[10]])
        corr148, p148 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[11]])
        corr149, p149 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[12]])
        corr150, p150 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[13]])
        corr151, p151 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[14]])
        corr152, p152 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[15]])
        corr153, p153 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[16]])
        corr154, p154 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[0]])
        corr155, p155 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[1]])
        corr156, p156 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[2]])
        corr157, p157 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[3]])
        corr158, p158 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[4]])
        corr159, p159 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[5]])
        corr160, p160 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[6]])
        corr161, p161 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[7]])
        corr162, p162 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[8]])
        corr163, p163 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[9]])
        corr164, p164 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[10]])
        corr165, p165 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[11]])
        corr166, p166 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[12]])
        corr167, p167 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[13]])
        corr168, p168 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[14]])
        corr169, p169 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[15]])
        corr170, p170 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[16]])
        corr171, p171 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[17]])


        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        # plot XLVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        # plot XLVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        # plot XLVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        # plot XLIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        # plot L
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        # plot LI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        # plot LII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        # plot LIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        # plot LIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        # plot LV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        # plot LVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        # plot LVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        # plot LVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        # plot LIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        # plot LX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        # plot LXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        # plot LXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        # plot LXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        # plot LXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        # plot LXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        # plot LXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        # plot LXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        # plot LXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        # plot LXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        # plot LXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        # plot LXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        # plot LXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        # plot LXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        # plot LXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        # plot LXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        # plot LXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        # plot LXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        # plot LXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

        # plot LXXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr79, 3)} | P-value: {round(p79, 4)} \n -----------------');

        # plot LXXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr80, 3)} | P-value: {round(p80, 4)} \n -----------------');

        # plot LXXXI
        sns.lmplot(x = quant_vars[3], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr81, 3)} | P-value: {round(p81, 4)} \n -----------------');

        # plot LXXXII
        sns.lmplot(x = quant_vars[4], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr82, 3)} | P-value: {round(p82, 4)} \n -----------------');

        # plot LXXXIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr83, 3)} | P-value: {round(p83, 4)} \n -----------------');

        # plot LXXXIV
        sns.lmplot(x = quant_vars[6], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr84, 3)} | P-value: {round(p84, 4)} \n -----------------');

        # plot LXXXV
        sns.lmplot(x = quant_vars[7], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr85, 3)} | P-value: {round(p85, 4)} \n -----------------');

        # plot LXXXVI
        sns.lmplot(x = quant_vars[8], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr86, 3)} | P-value: {round(p86, 4)} \n -----------------');

        # plot LXXXVII
        sns.lmplot(x = quant_vars[9], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr87, 3)} | P-value: {round(p87, 4)} \n -----------------');

        # plot LXXXVIII
        sns.lmplot(x = quant_vars[10], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr88, 3)} | P-value: {round(p88, 4)} \n -----------------');

        # plot LXXXIX
        sns.lmplot(x = quant_vars[11], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr89, 3)} | P-value: {round(p89, 4)} \n -----------------');

        # plot XC
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr90, 3)} | P-value: {round(p90, 4)} \n -----------------');

        # plot XCI
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr91, 3)} | P-value: {round(p91, 4)} \n -----------------');

        # plot XCII
        sns.lmplot(x = quant_vars[0], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr92, 3)} | P-value: {round(p92, 4)} \n -----------------');

        # plot XCIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr93, 3)} | P-value: {round(p93, 4)} \n -----------------');

        # plot XCIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr94, 3)} | P-value: {round(p94, 4)} \n -----------------');

        # plot XCV
        sns.lmplot(x = quant_vars[3], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr95, 3)} | P-value: {round(p95, 4)} \n -----------------');

        # plot XCVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr96, 3)} | P-value: {round(p96, 4)} \n -----------------');

        # plot XCVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr97, 3)} | P-value: {round(p97, 4)} \n -----------------');

        # plot XCVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr98, 3)} | P-value: {round(p98, 4)} \n -----------------');

        # plot XCIX
        sns.lmplot(x = quant_vars[7], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr99, 3)} | P-value: {round(p99, 4)} \n -----------------');

        # plot C
        sns.lmplot(x = quant_vars[8], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr100, 3)} | P-value: {round(p100, 4)} \n -----------------');

        # plot CI
        sns.lmplot(x = quant_vars[9], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr101, 3)} | P-value: {round(p101, 4)} \n -----------------');

        # plot CII
        sns.lmplot(x = quant_vars[10], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr102, 3)} | P-value: {round(p102, 4)} \n -----------------');

        # plot CIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr103, 3)} | P-value: {round(p103, 4)} \n -----------------');

        # plot CIV
        sns.lmplot(x = quant_vars[12], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr104, 3)} | P-value: {round(p104, 4)} \n -----------------');

        # plot CV
        sns.lmplot(x = quant_vars[13], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr105, 3)} | P-value: {round(p105, 4)} \n -----------------');

        # plot CVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr106, 3)} | P-value: {round(p106, 4)} \n -----------------');

        # plot CVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr107, 3)} | P-value: {round(p107, 4)} \n -----------------');

        # plot CVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr108, 3)} | P-value: {round(p108, 4)} \n -----------------');

        # plot CVIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr109, 3)} | P-value: {round(p109, 4)} \n -----------------');

        # plot CX
        sns.lmplot(x = quant_vars[4], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr110, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr111, 3)} | P-value: {round(p111, 4)} \n -----------------');

        # plot CXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr112, 3)} | P-value: {round(p112, 4)} \n -----------------');

        # plot CXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr113, 3)} | P-value: {round(p113, 4)} \n -----------------');

        # plot CXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr114, 3)} | P-value: {round(p114, 4)} \n -----------------');

        # plot CXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr115, 3)} | P-value: {round(p115, 4)} \n -----------------');

        # plot CXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr116, 3)} | P-value: {round(p116, 4)} \n -----------------');

        # plot CXVII
        sns.lmplot(x = quant_vars[11], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr117, 3)} | P-value: {round(p117, 4)} \n -----------------');

        # plot CXVIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr118, 3)} | P-value: {round(p118, 4)} \n -----------------');

        # plot CXIX
        sns.lmplot(x = quant_vars[13], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr119, 3)} | P-value: {round(p119, 4)} \n -----------------');

        # plot CXX
        sns.lmplot(x = quant_vars[14], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr120, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXXI
        sns.lmplot(x = quant_vars[0], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr121, 3)} | P-value: {round(p121, 4)} \n -----------------');

        # plot CXXII
        sns.lmplot(x = quant_vars[1], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr122, 3)} | P-value: {round(p122, 4)} \n -----------------');

        # plot CXXIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr123, 3)} | P-value: {round(p123, 4)} \n -----------------');

        # plot CXXIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr124, 3)} | P-value: {round(p124, 4)} \n -----------------');

        # plot CXXV
        sns.lmplot(x = quant_vars[4], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr125, 3)} | P-value: {round(p125, 4)} \n -----------------');

        # plot CXXVI
        sns.lmplot(x = quant_vars[5], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr126, 3)} | P-value: {round(p126, 4)} \n -----------------');

        # plot CXXVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr127, 3)} | P-value: {round(p127, 4)} \n -----------------');

        # plot CXXVIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr128, 3)} | P-value: {round(p128, 4)} \n -----------------');

        # plot CXXIX
        sns.lmplot(x = quant_vars[8], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr129, 3)} | P-value: {round(p129, 4)} \n -----------------');

        # plot CXXX
        sns.lmplot(x = quant_vars[9], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr130, 3)} | P-value: {round(p130, 4)} \n -----------------');

        # plot CXXXI
        sns.lmplot(x = quant_vars[10], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr131, 3)} | P-value: {round(p131, 4)} \n -----------------');

        # plot CXXXII
        sns.lmplot(x = quant_vars[11], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr132, 3)} | P-value: {round(p132, 4)} \n -----------------');

        # plot CXXXIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr133, 3)} | P-value: {round(p133, 4)} \n -----------------');

        # plot CXXXIV
        sns.lmplot(x = quant_vars[13], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr134, 3)} | P-value: {round(p134, 4)} \n -----------------');

        # plot CXXXV
        sns.lmplot(x = quant_vars[14], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr135, 3)} | P-value: {round(p135, 4)} \n -----------------');

        # plot CXXXVI
        sns.lmplot(x = quant_vars[15], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr136, 3)} | P-value: {round(p136, 4)} \n -----------------');

        # plot CXXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr137, 3)} | P-value: {round(p137, 4)} \n -----------------');

        # plot CXXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr138, 3)} | P-value: {round(p138, 4)} \n -----------------');

        # plot CXXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr139, 3)} | P-value: {round(p139, 4)} \n -----------------');

        # plot CXL
        sns.lmplot(x = quant_vars[3], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr140, 3)} | P-value: {round(p140, 4)} \n -----------------');

        # plot CXLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr141, 3)} | P-value: {round(p141, 4)} \n -----------------');

        # plot CXLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr142, 3)} | P-value: {round(p142, 4)} \n -----------------');

        # plot CXLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr143, 3)} | P-value: {round(p143, 4)} \n -----------------');

        # plot CXLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr144, 3)} | P-value: {round(p144, 4)} \n -----------------');

        # plot CXLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr145, 3)} | P-value: {round(p145, 4)} \n -----------------');

        # plot CXLVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr146, 3)} | P-value: {round(p146, 4)} \n -----------------');

        # plot CXLVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr147, 3)} | P-value: {round(p147, 4)} \n -----------------');

        # plot CXLVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr148, 3)} | P-value: {round(p148, 4)} \n -----------------');

        # plot CXLIX
        sns.lmplot(x = quant_vars[12], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr149, 3)} | P-value: {round(p149, 4)} \n -----------------');

        # plot CL
        sns.lmplot(x = quant_vars[13], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr150, 3)} | P-value: {round(p150, 4)} \n -----------------');

        # plot CLI
        sns.lmplot(x = quant_vars[14], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr151, 3)} | P-value: {round(p151, 4)} \n -----------------');

        # plot CLII
        sns.lmplot(x = quant_vars[15], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr152, 3)} | P-value: {round(p152, 4)} \n -----------------');

        # plot CLIII
        sns.lmplot(x = quant_vars[16], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr153, 3)} | P-value: {round(p153, 4)} \n -----------------');

        # plot CLIV
        sns.lmplot(x = quant_vars[0], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr154, 3)} | P-value: {round(p154, 4)} \n -----------------');

        # plot CLV
        sns.lmplot(x = quant_vars[1], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr155, 3)} | P-value: {round(p155, 4)} \n -----------------');

        # plot CLVI
        sns.lmplot(x = quant_vars[2], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr156, 3)} | P-value: {round(p156, 4)} \n -----------------');

        # plot CLVII
        sns.lmplot(x = quant_vars[3], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr157, 3)} | P-value: {round(p157, 4)} \n -----------------');

        # plot CLVIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr158, 3)} | P-value: {round(p158, 4)} \n -----------------');

        # plot CLIX
        sns.lmplot(x = quant_vars[5], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr159, 3)} | P-value: {round(p159, 4)} \n -----------------');

        # plot CLX
        sns.lmplot(x = quant_vars[6], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr160, 3)} | P-value: {round(p160, 4)} \n -----------------');

        # plot CLXI
        sns.lmplot(x = quant_vars[7], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr161, 3)} | P-value: {round(p161, 4)} \n -----------------');

        # plot CLXII
        sns.lmplot(x = quant_vars[8], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr162, 3)} | P-value: {round(p162, 4)} \n -----------------');

        # plot CLXIII
        sns.lmplot(x = quant_vars[9], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr163, 3)} | P-value: {round(p163, 4)} \n -----------------');

        # plot CLXIV
        sns.lmplot(x = quant_vars[10], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr164, 3)} | P-value: {round(p164, 4)} \n -----------------');

        # plot CLXV
        sns.lmplot(x = quant_vars[11], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr165, 3)} | P-value: {round(p165, 4)} \n -----------------');

        # plot CLXVI
        sns.lmplot(x = quant_vars[12], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr166, 3)} | P-value: {round(p166, 4)} \n -----------------');

        # plot CLXVII
        sns.lmplot(x = quant_vars[13], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr167, 3)} | P-value: {round(p167, 4)} \n -----------------');

        # plot CLXVIII
        sns.lmplot(x = quant_vars[14], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr168, 3)} | P-value: {round(p168, 4)} \n -----------------');

        # plot CLXIX
        sns.lmplot(x = quant_vars[15], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr169, 3)} | P-value: {round(p169, 4)} \n -----------------');

        # plot CLXX
        sns.lmplot(x = quant_vars[16], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr170, 3)} | P-value: {round(p170, 4)} \n -----------------');

        # plot CLXXI
        sns.lmplot(x = quant_vars[17], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr171, 3)} | P-value: {round(p171, 4)} \n -----------------');


    if k == 20:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])
        corr79, p79 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[0]])
        corr80, p80 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[1]])
        corr81, p81 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[2]])
        corr82, p82 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[3]])
        corr83, p83 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[4]])
        corr84, p84 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[5]])
        corr85, p85 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[6]])
        corr86, p86 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[7]])
        corr87, p87 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[8]])
        corr88, p88 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[9]])
        corr89, p89 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[10]])
        corr90, p90 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[11]])
        corr91, p91 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[12]])
        corr92, p92 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[0]])
        corr93, p93 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[1]])
        corr94, p94 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[2]])
        corr95, p95 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[3]])
        corr96, p96 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[4]])
        corr97, p97 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[5]])
        corr98, p98 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[6]])
        corr99, p99 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[7]])
        corr100, p100 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[8]])
        corr101, p101 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[9]])
        corr102, p102 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[10]])
        corr103, p103 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[11]])
        corr104, p104 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[12]])
        corr105, p105 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[13]])
        corr106, p106 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[0]])
        corr107, p107 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[1]])
        corr108, p108 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[2]])
        corr109, p109 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[3]])
        corr110, p110 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[4]])
        corr111, p111 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[5]])
        corr112, p112 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[6]])
        corr113, p113 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[7]])
        corr114, p114 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[8]])
        corr115, p115 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[9]])
        corr116, p116 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[10]])
        corr117, p117 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[11]])
        corr118, p118 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[12]])
        corr119, p119 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[13]])
        corr120, p120 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[14]])
        corr121, p121 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[0]])
        corr122, p122 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[1]])
        corr123, p123 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[2]])
        corr124, p124 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[3]])
        corr125, p125 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[4]])
        corr126, p126 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[5]])
        corr127, p127 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[6]])
        corr128, p128 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[7]])
        corr129, p129 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[8]])
        corr130, p130 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[9]])
        corr131, p131 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[10]])
        corr132, p132 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[11]])
        corr133, p133 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[12]])
        corr134, p134 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[13]])
        corr135, p135 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[14]])
        corr136, p136 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[15]])
        corr137, p137 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[0]])
        corr138, p138 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[1]])
        corr139, p139 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[2]])
        corr140, p140 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[3]])
        corr141, p141 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[4]])
        corr142, p142 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[5]])
        corr143, p143 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[6]])
        corr144, p144 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[7]])
        corr145, p145 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[8]])
        corr146, p146 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[9]])
        corr147, p147 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[10]])
        corr148, p148 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[11]])
        corr149, p149 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[12]])
        corr150, p150 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[13]])
        corr151, p151 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[14]])
        corr152, p152 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[15]])
        corr153, p153 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[16]])
        corr154, p154 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[0]])
        corr155, p155 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[1]])
        corr156, p156 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[2]])
        corr157, p157 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[3]])
        corr158, p158 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[4]])
        corr159, p159 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[5]])
        corr160, p160 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[6]])
        corr161, p161 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[7]])
        corr162, p162 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[8]])
        corr163, p163 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[9]])
        corr164, p164 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[10]])
        corr165, p165 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[11]])
        corr166, p166 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[12]])
        corr167, p167 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[13]])
        corr168, p168 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[14]])
        corr169, p169 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[15]])
        corr170, p170 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[16]])
        corr171, p171 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[17]])
        corr172, p172 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[0]])
        corr173, p173 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[1]])
        corr174, p174 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[2]])
        corr175, p175 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[3]])
        corr176, p176 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[4]])
        corr177, p177 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[5]])
        corr178, p178 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[6]])
        corr179, p179 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[7]])
        corr180, p180 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[8]])
        corr181, p181 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[9]])
        corr182, p182 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[10]])
        corr183, p183 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[11]])
        corr184, p184 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[12]])
        corr185, p185 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[13]])
        corr186, p186 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[14]])
        corr187, p187 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[15]])
        corr188, p188 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[16]])
        corr189, p189 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[17]])
        corr190, p190 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[18]])

        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        # plot XLVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        # plot XLVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        # plot XLVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        # plot XLIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        # plot L
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        # plot LI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        # plot LII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        # plot LIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        # plot LIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        # plot LV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        # plot LVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        # plot LVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        # plot LVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        # plot LIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        # plot LX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        # plot LXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        # plot LXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        # plot LXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        # plot LXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        # plot LXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        # plot LXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        # plot LXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        # plot LXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        # plot LXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        # plot LXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        # plot LXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        # plot LXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        # plot LXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        # plot LXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        # plot LXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        # plot LXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        # plot LXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        # plot LXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

        # plot LXXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr79, 3)} | P-value: {round(p79, 4)} \n -----------------');

        # plot LXXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr80, 3)} | P-value: {round(p80, 4)} \n -----------------');

        # plot LXXXI
        sns.lmplot(x = quant_vars[3], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr81, 3)} | P-value: {round(p81, 4)} \n -----------------');

        # plot LXXXII
        sns.lmplot(x = quant_vars[4], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr82, 3)} | P-value: {round(p82, 4)} \n -----------------');

        # plot LXXXIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr83, 3)} | P-value: {round(p83, 4)} \n -----------------');

        # plot LXXXIV
        sns.lmplot(x = quant_vars[6], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr84, 3)} | P-value: {round(p84, 4)} \n -----------------');

        # plot LXXXV
        sns.lmplot(x = quant_vars[7], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr85, 3)} | P-value: {round(p85, 4)} \n -----------------');

        # plot LXXXVI
        sns.lmplot(x = quant_vars[8], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr86, 3)} | P-value: {round(p86, 4)} \n -----------------');

        # plot LXXXVII
        sns.lmplot(x = quant_vars[9], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr87, 3)} | P-value: {round(p87, 4)} \n -----------------');

        # plot LXXXVIII
        sns.lmplot(x = quant_vars[10], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr88, 3)} | P-value: {round(p88, 4)} \n -----------------');

        # plot LXXXIX
        sns.lmplot(x = quant_vars[11], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr89, 3)} | P-value: {round(p89, 4)} \n -----------------');

        # plot XC
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr90, 3)} | P-value: {round(p90, 4)} \n -----------------');

        # plot XCI
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr91, 3)} | P-value: {round(p91, 4)} \n -----------------');

        # plot XCII
        sns.lmplot(x = quant_vars[0], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr92, 3)} | P-value: {round(p92, 4)} \n -----------------');

        # plot XCIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr93, 3)} | P-value: {round(p93, 4)} \n -----------------');

        # plot XCIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr94, 3)} | P-value: {round(p94, 4)} \n -----------------');

        # plot XCV
        sns.lmplot(x = quant_vars[3], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr95, 3)} | P-value: {round(p95, 4)} \n -----------------');

        # plot XCVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr96, 3)} | P-value: {round(p96, 4)} \n -----------------');

        # plot XCVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr97, 3)} | P-value: {round(p97, 4)} \n -----------------');

        # plot XCVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr98, 3)} | P-value: {round(p98, 4)} \n -----------------');

        # plot XCIX
        sns.lmplot(x = quant_vars[7], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr99, 3)} | P-value: {round(p99, 4)} \n -----------------');

        # plot C
        sns.lmplot(x = quant_vars[8], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr100, 3)} | P-value: {round(p100, 4)} \n -----------------');

        # plot CI
        sns.lmplot(x = quant_vars[9], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr101, 3)} | P-value: {round(p101, 4)} \n -----------------');

        # plot CII
        sns.lmplot(x = quant_vars[10], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr102, 3)} | P-value: {round(p102, 4)} \n -----------------');

        # plot CIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr103, 3)} | P-value: {round(p103, 4)} \n -----------------');

        # plot CIV
        sns.lmplot(x = quant_vars[12], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr104, 3)} | P-value: {round(p104, 4)} \n -----------------');

        # plot CV
        sns.lmplot(x = quant_vars[13], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr105, 3)} | P-value: {round(p105, 4)} \n -----------------');

        # plot CVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr106, 3)} | P-value: {round(p106, 4)} \n -----------------');

        # plot CVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr107, 3)} | P-value: {round(p107, 4)} \n -----------------');

        # plot CVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr108, 3)} | P-value: {round(p108, 4)} \n -----------------');

        # plot CVIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr109, 3)} | P-value: {round(p109, 4)} \n -----------------');

        # plot CX
        sns.lmplot(x = quant_vars[4], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr110, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr111, 3)} | P-value: {round(p111, 4)} \n -----------------');

        # plot CXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr112, 3)} | P-value: {round(p112, 4)} \n -----------------');

        # plot CXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr113, 3)} | P-value: {round(p113, 4)} \n -----------------');

        # plot CXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr114, 3)} | P-value: {round(p114, 4)} \n -----------------');

        # plot CXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr115, 3)} | P-value: {round(p115, 4)} \n -----------------');

        # plot CXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr116, 3)} | P-value: {round(p116, 4)} \n -----------------');

        # plot CXVII
        sns.lmplot(x = quant_vars[11], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr117, 3)} | P-value: {round(p117, 4)} \n -----------------');

        # plot CXVIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr118, 3)} | P-value: {round(p118, 4)} \n -----------------');

        # plot CXIX
        sns.lmplot(x = quant_vars[13], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr119, 3)} | P-value: {round(p119, 4)} \n -----------------');

        # plot CXX
        sns.lmplot(x = quant_vars[14], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr120, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXXI
        sns.lmplot(x = quant_vars[0], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr121, 3)} | P-value: {round(p121, 4)} \n -----------------');

        # plot CXXII
        sns.lmplot(x = quant_vars[1], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr122, 3)} | P-value: {round(p122, 4)} \n -----------------');

        # plot CXXIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr123, 3)} | P-value: {round(p123, 4)} \n -----------------');

        # plot CXXIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr124, 3)} | P-value: {round(p124, 4)} \n -----------------');

        # plot CXXV
        sns.lmplot(x = quant_vars[4], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr125, 3)} | P-value: {round(p125, 4)} \n -----------------');

        # plot CXXVI
        sns.lmplot(x = quant_vars[5], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr126, 3)} | P-value: {round(p126, 4)} \n -----------------');

        # plot CXXVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr127, 3)} | P-value: {round(p127, 4)} \n -----------------');

        # plot CXXVIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr128, 3)} | P-value: {round(p128, 4)} \n -----------------');

        # plot CXXIX
        sns.lmplot(x = quant_vars[8], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr129, 3)} | P-value: {round(p129, 4)} \n -----------------');

        # plot CXXX
        sns.lmplot(x = quant_vars[9], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr130, 3)} | P-value: {round(p130, 4)} \n -----------------');

        # plot CXXXI
        sns.lmplot(x = quant_vars[10], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr131, 3)} | P-value: {round(p131, 4)} \n -----------------');

        # plot CXXXII
        sns.lmplot(x = quant_vars[11], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr132, 3)} | P-value: {round(p132, 4)} \n -----------------');

        # plot CXXXIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr133, 3)} | P-value: {round(p133, 4)} \n -----------------');

        # plot CXXXIV
        sns.lmplot(x = quant_vars[13], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr134, 3)} | P-value: {round(p134, 4)} \n -----------------');

        # plot CXXXV
        sns.lmplot(x = quant_vars[14], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr135, 3)} | P-value: {round(p135, 4)} \n -----------------');

        # plot CXXXVI
        sns.lmplot(x = quant_vars[15], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr136, 3)} | P-value: {round(p136, 4)} \n -----------------');

        # plot CXXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr137, 3)} | P-value: {round(p137, 4)} \n -----------------');

        # plot CXXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr138, 3)} | P-value: {round(p138, 4)} \n -----------------');

        # plot CXXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr139, 3)} | P-value: {round(p139, 4)} \n -----------------');

        # plot CXL
        sns.lmplot(x = quant_vars[3], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr140, 3)} | P-value: {round(p140, 4)} \n -----------------');

        # plot CXLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr141, 3)} | P-value: {round(p141, 4)} \n -----------------');

        # plot CXLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr142, 3)} | P-value: {round(p142, 4)} \n -----------------');

        # plot CXLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr143, 3)} | P-value: {round(p143, 4)} \n -----------------');

        # plot CXLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr144, 3)} | P-value: {round(p144, 4)} \n -----------------');

        # plot CXLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr145, 3)} | P-value: {round(p145, 4)} \n -----------------');

        # plot CXLVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr146, 3)} | P-value: {round(p146, 4)} \n -----------------');

        # plot CXLVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr147, 3)} | P-value: {round(p147, 4)} \n -----------------');

        # plot CXLVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr148, 3)} | P-value: {round(p148, 4)} \n -----------------');

        # plot CXLIX
        sns.lmplot(x = quant_vars[12], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr149, 3)} | P-value: {round(p149, 4)} \n -----------------');

        # plot CL
        sns.lmplot(x = quant_vars[13], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr150, 3)} | P-value: {round(p150, 4)} \n -----------------');

        # plot CLI
        sns.lmplot(x = quant_vars[14], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr151, 3)} | P-value: {round(p151, 4)} \n -----------------');

        # plot CLII
        sns.lmplot(x = quant_vars[15], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr152, 3)} | P-value: {round(p152, 4)} \n -----------------');

        # plot CLIII
        sns.lmplot(x = quant_vars[16], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr153, 3)} | P-value: {round(p153, 4)} \n -----------------');

        # plot CLIV
        sns.lmplot(x = quant_vars[0], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr154, 3)} | P-value: {round(p154, 4)} \n -----------------');

        # plot CLV
        sns.lmplot(x = quant_vars[1], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr155, 3)} | P-value: {round(p155, 4)} \n -----------------');

        # plot CLVI
        sns.lmplot(x = quant_vars[2], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr156, 3)} | P-value: {round(p156, 4)} \n -----------------');

        # plot CLVII
        sns.lmplot(x = quant_vars[3], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr157, 3)} | P-value: {round(p157, 4)} \n -----------------');

        # plot CLVIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr158, 3)} | P-value: {round(p158, 4)} \n -----------------');

        # plot CLIX
        sns.lmplot(x = quant_vars[5], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr159, 3)} | P-value: {round(p159, 4)} \n -----------------');

        # plot CLX
        sns.lmplot(x = quant_vars[6], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr160, 3)} | P-value: {round(p160, 4)} \n -----------------');

        # plot CLXI
        sns.lmplot(x = quant_vars[7], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr161, 3)} | P-value: {round(p161, 4)} \n -----------------');

        # plot CLXII
        sns.lmplot(x = quant_vars[8], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr162, 3)} | P-value: {round(p162, 4)} \n -----------------');

        # plot CLXIII
        sns.lmplot(x = quant_vars[9], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr163, 3)} | P-value: {round(p163, 4)} \n -----------------');

        # plot CLXIV
        sns.lmplot(x = quant_vars[10], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr164, 3)} | P-value: {round(p164, 4)} \n -----------------');

        # plot CLXV
        sns.lmplot(x = quant_vars[11], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr165, 3)} | P-value: {round(p165, 4)} \n -----------------');

        # plot CLXVI
        sns.lmplot(x = quant_vars[12], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr166, 3)} | P-value: {round(p166, 4)} \n -----------------');

        # plot CLXVII
        sns.lmplot(x = quant_vars[13], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr167, 3)} | P-value: {round(p167, 4)} \n -----------------');

        # plot CLXVIII
        sns.lmplot(x = quant_vars[14], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr168, 3)} | P-value: {round(p168, 4)} \n -----------------');

        # plot CLXIX
        sns.lmplot(x = quant_vars[15], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr169, 3)} | P-value: {round(p169, 4)} \n -----------------');

        # plot CLXX
        sns.lmplot(x = quant_vars[16], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr170, 3)} | P-value: {round(p170, 4)} \n -----------------');

        # plot CLXXI
        sns.lmplot(x = quant_vars[17], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr171, 3)} | P-value: {round(p171, 4)} \n -----------------');

        # plot CLXXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr172, 3)} | P-value: {round(p172, 4)} \n -----------------');

        # plot CLXXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr173, 3)} | P-value: {round(p173, 4)} \n -----------------');

        # plot CLXXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr174, 3)} | P-value: {round(p174, 4)} \n -----------------');

        # plot CLXXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr175, 3)} | P-value: {round(p175, 4)} \n -----------------');

        # plot CLXXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr176, 3)} | P-value: {round(p176, 4)} \n -----------------');

        # plot CLXXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr177, 3)} | P-value: {round(p177, 4)} \n -----------------');

        # plot CLXXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr178, 3)} | P-value: {round(p178, 4)} \n -----------------');

        # plot CLXXIX
        sns.lmplot(x = quant_vars[7], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr179, 3)} | P-value: {round(p179, 4)} \n -----------------');

        # plot CLXXX
        sns.lmplot(x = quant_vars[8], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr180, 3)} | P-value: {round(p180, 4)} \n -----------------');

        # plot CLXXXI
        sns.lmplot(x = quant_vars[9], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr181, 3)} | P-value: {round(p181, 4)} \n -----------------');

        # plot CLXXXII
        sns.lmplot(x = quant_vars[10], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr182, 3)} | P-value: {round(p182, 4)} \n -----------------');

        # plot CLXXXIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr183, 3)} | P-value: {round(p183, 4)} \n -----------------');

        # plot CLXXXIV
        sns.lmplot(x = quant_vars[12], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr184, 3)} | P-value: {round(p184, 4)} \n -----------------');

        # plot CLXXXV
        sns.lmplot(x = quant_vars[13], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr185, 3)} | P-value: {round(p185, 4)} \n -----------------');

        # plot CLXXXVI
        sns.lmplot(x = quant_vars[14], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr186, 3)} | P-value: {round(p186, 4)} \n -----------------');

        # plot CLXXXVII
        sns.lmplot(x = quant_vars[15], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr187, 3)} | P-value: {round(p187, 4)} \n -----------------');

        # plot CLXXXVIII
        sns.lmplot(x = quant_vars[16], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr188, 3)} | P-value: {round(p188, 4)} \n -----------------');

        # plot CLXXXIX
        sns.lmplot(x = quant_vars[17], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr189, 3)} | P-value: {round(p189, 4)} \n -----------------');

        # plot CXC
        sns.lmplot(x = quant_vars[18], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr190, 3)} | P-value: {round(p190, 4)} \n -----------------');

    if k == 21:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])
        corr79, p79 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[0]])
        corr80, p80 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[1]])
        corr81, p81 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[2]])
        corr82, p82 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[3]])
        corr83, p83 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[4]])
        corr84, p84 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[5]])
        corr85, p85 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[6]])
        corr86, p86 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[7]])
        corr87, p87 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[8]])
        corr88, p88 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[9]])
        corr89, p89 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[10]])
        corr90, p90 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[11]])
        corr91, p91 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[12]])
        corr92, p92 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[0]])
        corr93, p93 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[1]])
        corr94, p94 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[2]])
        corr95, p95 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[3]])
        corr96, p96 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[4]])
        corr97, p97 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[5]])
        corr98, p98 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[6]])
        corr99, p99 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[7]])
        corr100, p100 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[8]])
        corr101, p101 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[9]])
        corr102, p102 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[10]])
        corr103, p103 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[11]])
        corr104, p104 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[12]])
        corr105, p105 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[13]])
        corr106, p106 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[0]])
        corr107, p107 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[1]])
        corr108, p108 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[2]])
        corr109, p109 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[3]])
        corr110, p110 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[4]])
        corr111, p111 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[5]])
        corr112, p112 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[6]])
        corr113, p113 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[7]])
        corr114, p114 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[8]])
        corr115, p115 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[9]])
        corr116, p116 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[10]])
        corr117, p117 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[11]])
        corr118, p118 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[12]])
        corr119, p119 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[13]])
        corr120, p120 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[14]])
        corr121, p121 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[0]])
        corr122, p122 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[1]])
        corr123, p123 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[2]])
        corr124, p124 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[3]])
        corr125, p125 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[4]])
        corr126, p126 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[5]])
        corr127, p127 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[6]])
        corr128, p128 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[7]])
        corr129, p129 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[8]])
        corr130, p130 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[9]])
        corr131, p131 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[10]])
        corr132, p132 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[11]])
        corr133, p133 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[12]])
        corr134, p134 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[13]])
        corr135, p135 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[14]])
        corr136, p136 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[15]])
        corr137, p137 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[0]])
        corr138, p138 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[1]])
        corr139, p139 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[2]])
        corr140, p140 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[3]])
        corr141, p141 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[4]])
        corr142, p142 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[5]])
        corr143, p143 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[6]])
        corr144, p144 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[7]])
        corr145, p145 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[8]])
        corr146, p146 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[9]])
        corr147, p147 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[10]])
        corr148, p148 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[11]])
        corr149, p149 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[12]])
        corr150, p150 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[13]])
        corr151, p151 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[14]])
        corr152, p152 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[15]])
        corr153, p153 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[16]])
        corr154, p154 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[0]])
        corr155, p155 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[1]])
        corr156, p156 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[2]])
        corr157, p157 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[3]])
        corr158, p158 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[4]])
        corr159, p159 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[5]])
        corr160, p160 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[6]])
        corr161, p161 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[7]])
        corr162, p162 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[8]])
        corr163, p163 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[9]])
        corr164, p164 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[10]])
        corr165, p165 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[11]])
        corr166, p166 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[12]])
        corr167, p167 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[13]])
        corr168, p168 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[14]])
        corr169, p169 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[15]])
        corr170, p170 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[16]])
        corr171, p171 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[17]])
        corr172, p172 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[0]])
        corr173, p173 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[1]])
        corr174, p174 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[2]])
        corr175, p175 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[3]])
        corr176, p176 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[4]])
        corr177, p177 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[5]])
        corr178, p178 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[6]])
        corr179, p179 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[7]])
        corr180, p180 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[8]])
        corr181, p181 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[9]])
        corr182, p182 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[10]])
        corr183, p183 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[11]])
        corr184, p184 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[12]])
        corr185, p185 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[13]])
        corr186, p186 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[14]])
        corr187, p187 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[15]])
        corr188, p188 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[16]])
        corr189, p189 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[17]])
        corr190, p190 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[18]])
        corr191, p191 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[0]])
        corr192, p192 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[1]])
        corr193, p193 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[2]])
        corr194, p194 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[3]])
        corr195, p195 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[4]])
        corr196, p196 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[5]])
        corr197, p197 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[6]])
        corr198, p198 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[7]])
        corr199, p199 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[8]])
        corr200, p200 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[9]])
        corr201, p201 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[10]])
        corr202, p202 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[11]])
        corr203, p203 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[12]])
        corr204, p204 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[13]])
        corr205, p205 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[14]])
        corr206, p206 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[15]])
        corr207, p207 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[16]])
        corr208, p208 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[17]])
        corr209, p209 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[18]])
        corr210, p210 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[19]])


        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        # plot XLVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        # plot XLVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        # plot XLVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        # plot XLIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        # plot L
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        # plot LI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        # plot LII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        # plot LIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        # plot LIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        # plot LV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        # plot LVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        # plot LVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        # plot LVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        # plot LIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        # plot LX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        # plot LXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        # plot LXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        # plot LXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        # plot LXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        # plot LXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        # plot LXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        # plot LXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        # plot LXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        # plot LXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        # plot LXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        # plot LXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        # plot LXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        # plot LXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        # plot LXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        # plot LXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        # plot LXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        # plot LXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        # plot LXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

        # plot LXXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr79, 3)} | P-value: {round(p79, 4)} \n -----------------');

        # plot LXXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr80, 3)} | P-value: {round(p80, 4)} \n -----------------');

        # plot LXXXI
        sns.lmplot(x = quant_vars[3], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr81, 3)} | P-value: {round(p81, 4)} \n -----------------');

        # plot LXXXII
        sns.lmplot(x = quant_vars[4], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr82, 3)} | P-value: {round(p82, 4)} \n -----------------');

        # plot LXXXIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr83, 3)} | P-value: {round(p83, 4)} \n -----------------');

        # plot LXXXIV
        sns.lmplot(x = quant_vars[6], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr84, 3)} | P-value: {round(p84, 4)} \n -----------------');

        # plot LXXXV
        sns.lmplot(x = quant_vars[7], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr85, 3)} | P-value: {round(p85, 4)} \n -----------------');

        # plot LXXXVI
        sns.lmplot(x = quant_vars[8], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr86, 3)} | P-value: {round(p86, 4)} \n -----------------');

        # plot LXXXVII
        sns.lmplot(x = quant_vars[9], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr87, 3)} | P-value: {round(p87, 4)} \n -----------------');

        # plot LXXXVIII
        sns.lmplot(x = quant_vars[10], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr88, 3)} | P-value: {round(p88, 4)} \n -----------------');

        # plot LXXXIX
        sns.lmplot(x = quant_vars[11], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr89, 3)} | P-value: {round(p89, 4)} \n -----------------');

        # plot XC
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr90, 3)} | P-value: {round(p90, 4)} \n -----------------');

        # plot XCI
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr91, 3)} | P-value: {round(p91, 4)} \n -----------------');

        # plot XCII
        sns.lmplot(x = quant_vars[0], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr92, 3)} | P-value: {round(p92, 4)} \n -----------------');

        # plot XCIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr93, 3)} | P-value: {round(p93, 4)} \n -----------------');

        # plot XCIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr94, 3)} | P-value: {round(p94, 4)} \n -----------------');

        # plot XCV
        sns.lmplot(x = quant_vars[3], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr95, 3)} | P-value: {round(p95, 4)} \n -----------------');

        # plot XCVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr96, 3)} | P-value: {round(p96, 4)} \n -----------------');

        # plot XCVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr97, 3)} | P-value: {round(p97, 4)} \n -----------------');

        # plot XCVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr98, 3)} | P-value: {round(p98, 4)} \n -----------------');

        # plot XCIX
        sns.lmplot(x = quant_vars[7], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr99, 3)} | P-value: {round(p99, 4)} \n -----------------');

        # plot C
        sns.lmplot(x = quant_vars[8], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr100, 3)} | P-value: {round(p100, 4)} \n -----------------');

        # plot CI
        sns.lmplot(x = quant_vars[9], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr101, 3)} | P-value: {round(p101, 4)} \n -----------------');

        # plot CII
        sns.lmplot(x = quant_vars[10], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr102, 3)} | P-value: {round(p102, 4)} \n -----------------');

        # plot CIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr103, 3)} | P-value: {round(p103, 4)} \n -----------------');

        # plot CIV
        sns.lmplot(x = quant_vars[12], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr104, 3)} | P-value: {round(p104, 4)} \n -----------------');

        # plot CV
        sns.lmplot(x = quant_vars[13], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr105, 3)} | P-value: {round(p105, 4)} \n -----------------');

        # plot CVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr106, 3)} | P-value: {round(p106, 4)} \n -----------------');

        # plot CVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr107, 3)} | P-value: {round(p107, 4)} \n -----------------');

        # plot CVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr108, 3)} | P-value: {round(p108, 4)} \n -----------------');

        # plot CVIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr109, 3)} | P-value: {round(p109, 4)} \n -----------------');

        # plot CX
        sns.lmplot(x = quant_vars[4], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr110, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr111, 3)} | P-value: {round(p111, 4)} \n -----------------');

        # plot CXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr112, 3)} | P-value: {round(p112, 4)} \n -----------------');

        # plot CXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr113, 3)} | P-value: {round(p113, 4)} \n -----------------');

        # plot CXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr114, 3)} | P-value: {round(p114, 4)} \n -----------------');

        # plot CXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr115, 3)} | P-value: {round(p115, 4)} \n -----------------');

        # plot CXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr116, 3)} | P-value: {round(p116, 4)} \n -----------------');

        # plot CXVII
        sns.lmplot(x = quant_vars[11], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr117, 3)} | P-value: {round(p117, 4)} \n -----------------');

        # plot CXVIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr118, 3)} | P-value: {round(p118, 4)} \n -----------------');

        # plot CXIX
        sns.lmplot(x = quant_vars[13], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr119, 3)} | P-value: {round(p119, 4)} \n -----------------');

        # plot CXX
        sns.lmplot(x = quant_vars[14], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr120, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXXI
        sns.lmplot(x = quant_vars[0], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr121, 3)} | P-value: {round(p121, 4)} \n -----------------');

        # plot CXXII
        sns.lmplot(x = quant_vars[1], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr122, 3)} | P-value: {round(p122, 4)} \n -----------------');

        # plot CXXIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr123, 3)} | P-value: {round(p123, 4)} \n -----------------');

        # plot CXXIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr124, 3)} | P-value: {round(p124, 4)} \n -----------------');

        # plot CXXV
        sns.lmplot(x = quant_vars[4], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr125, 3)} | P-value: {round(p125, 4)} \n -----------------');

        # plot CXXVI
        sns.lmplot(x = quant_vars[5], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr126, 3)} | P-value: {round(p126, 4)} \n -----------------');

        # plot CXXVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr127, 3)} | P-value: {round(p127, 4)} \n -----------------');

        # plot CXXVIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr128, 3)} | P-value: {round(p128, 4)} \n -----------------');

        # plot CXXIX
        sns.lmplot(x = quant_vars[8], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr129, 3)} | P-value: {round(p129, 4)} \n -----------------');

        # plot CXXX
        sns.lmplot(x = quant_vars[9], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr130, 3)} | P-value: {round(p130, 4)} \n -----------------');

        # plot CXXXI
        sns.lmplot(x = quant_vars[10], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr131, 3)} | P-value: {round(p131, 4)} \n -----------------');

        # plot CXXXII
        sns.lmplot(x = quant_vars[11], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr132, 3)} | P-value: {round(p132, 4)} \n -----------------');

        # plot CXXXIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr133, 3)} | P-value: {round(p133, 4)} \n -----------------');

        # plot CXXXIV
        sns.lmplot(x = quant_vars[13], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr134, 3)} | P-value: {round(p134, 4)} \n -----------------');

        # plot CXXXV
        sns.lmplot(x = quant_vars[14], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr135, 3)} | P-value: {round(p135, 4)} \n -----------------');

        # plot CXXXVI
        sns.lmplot(x = quant_vars[15], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr136, 3)} | P-value: {round(p136, 4)} \n -----------------');

        # plot CXXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr137, 3)} | P-value: {round(p137, 4)} \n -----------------');

        # plot CXXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr138, 3)} | P-value: {round(p138, 4)} \n -----------------');

        # plot CXXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr139, 3)} | P-value: {round(p139, 4)} \n -----------------');

        # plot CXL
        sns.lmplot(x = quant_vars[3], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr140, 3)} | P-value: {round(p140, 4)} \n -----------------');

        # plot CXLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr141, 3)} | P-value: {round(p141, 4)} \n -----------------');

        # plot CXLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr142, 3)} | P-value: {round(p142, 4)} \n -----------------');

        # plot CXLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr143, 3)} | P-value: {round(p143, 4)} \n -----------------');

        # plot CXLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr144, 3)} | P-value: {round(p144, 4)} \n -----------------');

        # plot CXLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr145, 3)} | P-value: {round(p145, 4)} \n -----------------');

        # plot CXLVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr146, 3)} | P-value: {round(p146, 4)} \n -----------------');

        # plot CXLVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr147, 3)} | P-value: {round(p147, 4)} \n -----------------');

        # plot CXLVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr148, 3)} | P-value: {round(p148, 4)} \n -----------------');

        # plot CXLIX
        sns.lmplot(x = quant_vars[12], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr149, 3)} | P-value: {round(p149, 4)} \n -----------------');

        # plot CL
        sns.lmplot(x = quant_vars[13], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr150, 3)} | P-value: {round(p150, 4)} \n -----------------');

        # plot CLI
        sns.lmplot(x = quant_vars[14], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr151, 3)} | P-value: {round(p151, 4)} \n -----------------');

        # plot CLII
        sns.lmplot(x = quant_vars[15], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr152, 3)} | P-value: {round(p152, 4)} \n -----------------');

        # plot CLIII
        sns.lmplot(x = quant_vars[16], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr153, 3)} | P-value: {round(p153, 4)} \n -----------------');

        # plot CLIV
        sns.lmplot(x = quant_vars[0], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr154, 3)} | P-value: {round(p154, 4)} \n -----------------');

        # plot CLV
        sns.lmplot(x = quant_vars[1], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr155, 3)} | P-value: {round(p155, 4)} \n -----------------');

        # plot CLVI
        sns.lmplot(x = quant_vars[2], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr156, 3)} | P-value: {round(p156, 4)} \n -----------------');

        # plot CLVII
        sns.lmplot(x = quant_vars[3], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr157, 3)} | P-value: {round(p157, 4)} \n -----------------');

        # plot CLVIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr158, 3)} | P-value: {round(p158, 4)} \n -----------------');

        # plot CLIX
        sns.lmplot(x = quant_vars[5], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr159, 3)} | P-value: {round(p159, 4)} \n -----------------');

        # plot CLX
        sns.lmplot(x = quant_vars[6], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr160, 3)} | P-value: {round(p160, 4)} \n -----------------');

        # plot CLXI
        sns.lmplot(x = quant_vars[7], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr161, 3)} | P-value: {round(p161, 4)} \n -----------------');

        # plot CLXII
        sns.lmplot(x = quant_vars[8], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr162, 3)} | P-value: {round(p162, 4)} \n -----------------');

        # plot CLXIII
        sns.lmplot(x = quant_vars[9], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr163, 3)} | P-value: {round(p163, 4)} \n -----------------');

        # plot CLXIV
        sns.lmplot(x = quant_vars[10], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr164, 3)} | P-value: {round(p164, 4)} \n -----------------');

        # plot CLXV
        sns.lmplot(x = quant_vars[11], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr165, 3)} | P-value: {round(p165, 4)} \n -----------------');

        # plot CLXVI
        sns.lmplot(x = quant_vars[12], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr166, 3)} | P-value: {round(p166, 4)} \n -----------------');

        # plot CLXVII
        sns.lmplot(x = quant_vars[13], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr167, 3)} | P-value: {round(p167, 4)} \n -----------------');

        # plot CLXVIII
        sns.lmplot(x = quant_vars[14], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr168, 3)} | P-value: {round(p168, 4)} \n -----------------');

        # plot CLXIX
        sns.lmplot(x = quant_vars[15], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr169, 3)} | P-value: {round(p169, 4)} \n -----------------');

        # plot CLXX
        sns.lmplot(x = quant_vars[16], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr170, 3)} | P-value: {round(p170, 4)} \n -----------------');

        # plot CLXXI
        sns.lmplot(x = quant_vars[17], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr171, 3)} | P-value: {round(p171, 4)} \n -----------------');

        # plot CLXXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr172, 3)} | P-value: {round(p172, 4)} \n -----------------');

        # plot CLXXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr173, 3)} | P-value: {round(p173, 4)} \n -----------------');

        # plot CLXXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr174, 3)} | P-value: {round(p174, 4)} \n -----------------');

        # plot CLXXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr175, 3)} | P-value: {round(p175, 4)} \n -----------------');

        # plot CLXXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr176, 3)} | P-value: {round(p176, 4)} \n -----------------');

        # plot CLXXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr177, 3)} | P-value: {round(p177, 4)} \n -----------------');

        # plot CLXXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr178, 3)} | P-value: {round(p178, 4)} \n -----------------');

        # plot CLXXIX
        sns.lmplot(x = quant_vars[7], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr179, 3)} | P-value: {round(p179, 4)} \n -----------------');

        # plot CLXXX
        sns.lmplot(x = quant_vars[8], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr180, 3)} | P-value: {round(p180, 4)} \n -----------------');

        # plot CLXXXI
        sns.lmplot(x = quant_vars[9], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr181, 3)} | P-value: {round(p181, 4)} \n -----------------');

        # plot CLXXXII
        sns.lmplot(x = quant_vars[10], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr182, 3)} | P-value: {round(p182, 4)} \n -----------------');

        # plot CLXXXIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr183, 3)} | P-value: {round(p183, 4)} \n -----------------');

        # plot CLXXXIV
        sns.lmplot(x = quant_vars[12], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr184, 3)} | P-value: {round(p184, 4)} \n -----------------');

        # plot CLXXXV
        sns.lmplot(x = quant_vars[13], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr185, 3)} | P-value: {round(p185, 4)} \n -----------------');

        # plot CLXXXVI
        sns.lmplot(x = quant_vars[14], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr186, 3)} | P-value: {round(p186, 4)} \n -----------------');

        # plot CLXXXVII
        sns.lmplot(x = quant_vars[15], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr187, 3)} | P-value: {round(p187, 4)} \n -----------------');

        # plot CLXXXVIII
        sns.lmplot(x = quant_vars[16], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr188, 3)} | P-value: {round(p188, 4)} \n -----------------');

        # plot CLXXXIX
        sns.lmplot(x = quant_vars[17], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr189, 3)} | P-value: {round(p189, 4)} \n -----------------');

        # plot CXC
        sns.lmplot(x = quant_vars[18], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr190, 3)} | P-value: {round(p190, 4)} \n -----------------');

        # plot CXCI
        sns.lmplot(x = quant_vars[0], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr191, 3)} | P-value: {round(p191, 4)} \n -----------------');

        # plot CXCII
        sns.lmplot(x = quant_vars[1], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr192, 3)} | P-value: {round(p192, 4)} \n -----------------');

        # plot CXCIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr193, 3)} | P-value: {round(p193, 4)} \n -----------------');

        # plot CXCIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr194, 3)} | P-value: {round(p194, 4)} \n -----------------');

        # plot CXCV
        sns.lmplot(x = quant_vars[4], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr195, 3)} | P-value: {round(p195, 4)} \n -----------------');

        # plot CXCVI
        sns.lmplot(x = quant_vars[5], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr196, 3)} | P-value: {round(p196, 4)} \n -----------------');

        # plot CXCVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr197, 3)} | P-value: {round(p197, 4)} \n -----------------');

        # plot CXCVIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr198, 3)} | P-value: {round(p198, 4)} \n -----------------');

        # plot CXCIX
        sns.lmplot(x = quant_vars[8], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr199, 3)} | P-value: {round(p199, 4)} \n -----------------');

        # plot CC
        sns.lmplot(x = quant_vars[9], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr200, 3)} | P-value: {round(p200, 4)} \n -----------------');

        # plot CCI
        sns.lmplot(x = quant_vars[10], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr201, 3)} | P-value: {round(p201, 4)} \n -----------------');

        # plot CCII
        sns.lmplot(x = quant_vars[11], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr202, 3)} | P-value: {round(p202, 4)} \n -----------------');

        # plot CCIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr203, 3)} | P-value: {round(p203, 4)} \n -----------------');

        # plot CCIV
        sns.lmplot(x = quant_vars[13], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr204, 3)} | P-value: {round(p204, 4)} \n -----------------');

        # plot CCV
        sns.lmplot(x = quant_vars[14], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr205, 3)} | P-value: {round(p205, 4)} \n -----------------');

        # plot CCVI
        sns.lmplot(x = quant_vars[15], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr206, 3)} | P-value: {round(p206, 4)} \n -----------------');

        # plot CCVII
        sns.lmplot(x = quant_vars[16], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr207, 3)} | P-value: {round(p207, 4)} \n -----------------');

        # plot CCVIII
        sns.lmplot(x = quant_vars[17], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr208, 3)} | P-value: {round(p208, 4)} \n -----------------');

        # plot CCIX
        sns.lmplot(x = quant_vars[18], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr209, 3)} | P-value: {round(p209, 4)} \n -----------------');

        # plot CCX
        sns.lmplot(x = quant_vars[19], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr210, 3)} | P-value: {round(p210, 4)} \n -----------------');

    if k == 22:

        # determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])
        corr79, p79 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[0]])
        corr80, p80 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[1]])
        corr81, p81 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[2]])
        corr82, p82 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[3]])
        corr83, p83 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[4]])
        corr84, p84 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[5]])
        corr85, p85 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[6]])
        corr86, p86 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[7]])
        corr87, p87 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[8]])
        corr88, p88 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[9]])
        corr89, p89 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[10]])
        corr90, p90 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[11]])
        corr91, p91 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[12]])
        corr92, p92 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[0]])
        corr93, p93 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[1]])
        corr94, p94 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[2]])
        corr95, p95 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[3]])
        corr96, p96 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[4]])
        corr97, p97 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[5]])
        corr98, p98 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[6]])
        corr99, p99 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[7]])
        corr100, p100 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[8]])
        corr101, p101 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[9]])
        corr102, p102 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[10]])
        corr103, p103 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[11]])
        corr104, p104 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[12]])
        corr105, p105 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[13]])
        corr106, p106 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[0]])
        corr107, p107 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[1]])
        corr108, p108 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[2]])
        corr109, p109 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[3]])
        corr110, p110 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[4]])
        corr111, p111 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[5]])
        corr112, p112 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[6]])
        corr113, p113 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[7]])
        corr114, p114 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[8]])
        corr115, p115 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[9]])
        corr116, p116 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[10]])
        corr117, p117 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[11]])
        corr118, p118 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[12]])
        corr119, p119 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[13]])
        corr120, p120 = stats.pearsonr(df[quant_vars[15]], df[quant_vars[14]])
        corr121, p121 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[0]])
        corr122, p122 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[1]])
        corr123, p123 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[2]])
        corr124, p124 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[3]])
        corr125, p125 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[4]])
        corr126, p126 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[5]])
        corr127, p127 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[6]])
        corr128, p128 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[7]])
        corr129, p129 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[8]])
        corr130, p130 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[9]])
        corr131, p131 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[10]])
        corr132, p132 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[11]])
        corr133, p133 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[12]])
        corr134, p134 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[13]])
        corr135, p135 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[14]])
        corr136, p136 = stats.pearsonr(df[quant_vars[16]], df[quant_vars[15]])
        corr137, p137 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[0]])
        corr138, p138 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[1]])
        corr139, p139 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[2]])
        corr140, p140 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[3]])
        corr141, p141 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[4]])
        corr142, p142 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[5]])
        corr143, p143 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[6]])
        corr144, p144 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[7]])
        corr145, p145 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[8]])
        corr146, p146 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[9]])
        corr147, p147 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[10]])
        corr148, p148 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[11]])
        corr149, p149 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[12]])
        corr150, p150 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[13]])
        corr151, p151 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[14]])
        corr152, p152 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[15]])
        corr153, p153 = stats.pearsonr(df[quant_vars[17]], df[quant_vars[16]])
        corr154, p154 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[0]])
        corr155, p155 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[1]])
        corr156, p156 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[2]])
        corr157, p157 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[3]])
        corr158, p158 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[4]])
        corr159, p159 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[5]])
        corr160, p160 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[6]])
        corr161, p161 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[7]])
        corr162, p162 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[8]])
        corr163, p163 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[9]])
        corr164, p164 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[10]])
        corr165, p165 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[11]])
        corr166, p166 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[12]])
        corr167, p167 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[13]])
        corr168, p168 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[14]])
        corr169, p169 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[15]])
        corr170, p170 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[16]])
        corr171, p171 = stats.pearsonr(df[quant_vars[18]], df[quant_vars[17]])
        corr172, p172 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[0]])
        corr173, p173 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[1]])
        corr174, p174 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[2]])
        corr175, p175 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[3]])
        corr176, p176 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[4]])
        corr177, p177 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[5]])
        corr178, p178 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[6]])
        corr179, p179 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[7]])
        corr180, p180 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[8]])
        corr181, p181 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[9]])
        corr182, p182 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[10]])
        corr183, p183 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[11]])
        corr184, p184 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[12]])
        corr185, p185 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[13]])
        corr186, p186 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[14]])
        corr187, p187 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[15]])
        corr188, p188 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[16]])
        corr189, p189 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[17]])
        corr190, p190 = stats.pearsonr(df[quant_vars[19]], df[quant_vars[18]])
        corr191, p191 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[0]])
        corr192, p192 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[1]])
        corr193, p193 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[2]])
        corr194, p194 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[3]])
        corr195, p195 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[4]])
        corr196, p196 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[5]])
        corr197, p197 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[6]])
        corr198, p198 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[7]])
        corr199, p199 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[8]])
        corr200, p200 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[9]])
        corr201, p201 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[10]])
        corr202, p202 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[11]])
        corr203, p203 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[12]])
        corr204, p204 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[13]])
        corr205, p205 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[14]])
        corr206, p206 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[15]])
        corr207, p207 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[16]])
        corr208, p208 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[17]])
        corr209, p209 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[18]])
        corr210, p210 = stats.pearsonr(df[quant_vars[20]], df[quant_vars[19]])
        corr211, p211 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[0]])
        corr212, p212 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[1]])
        corr213, p213 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[2]])
        corr214, p214 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[3]])
        corr215, p215 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[4]])
        corr216, p216 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[5]])
        corr217, p217 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[6]])
        corr218, p218 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[7]])
        corr219, p219 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[8]])
        corr220, p220 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[9]])
        corr221, p221 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[10]])
        corr222, p222 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[11]])
        corr223, p223 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[12]])
        corr224, p224 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[13]])
        corr225, p225 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[14]])
        corr226, p226 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[15]])
        corr227, p227 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[16]])
        corr228, p228 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[17]])
        corr229, p229 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[18]])
        corr230, p230 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[19]])
        corr231, p231 = stats.pearsonr(df[quant_vars[21]], df[quant_vars[20]])
        
        # plot relationships between continuous variables

        # plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        # plot II
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        # plot III
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        # plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        # plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        # plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        # plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        # plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        # plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        # plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        # plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        # plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        # plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        # plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        # plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        # plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        # plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        # plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        # plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        # plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        # plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        # plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        # plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        # plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        # plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        # plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        # plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        # plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        # plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        # plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        # plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        # plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        # plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        # plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        # plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        # plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        # plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        # plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        # plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        # plot XL
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        # plot XLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        # plot XLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        # plot XLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        # plot XLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        # plot XLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        # plot XLVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        # plot XLVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        # plot XLVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        # plot XLIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        # plot L
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        # plot LI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        # plot LII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        # plot LIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        # plot LIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        # plot LV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        # plot LVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        # plot LVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        # plot LVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        # plot LIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        # plot LX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        # plot LXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        # plot LXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        # plot LXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        # plot LXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        # plot LXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        # plot LXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        # plot LXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        # plot LXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        # plot LXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        # plot LXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        # plot LXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        # plot LXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        # plot LXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        # plot LXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        # plot LXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        # plot LXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        # plot LXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        # plot LXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

        # plot LXXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr79, 3)} | P-value: {round(p79, 4)} \n -----------------');

        # plot LXXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr80, 3)} | P-value: {round(p80, 4)} \n -----------------');

        # plot LXXXI
        sns.lmplot(x = quant_vars[3], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr81, 3)} | P-value: {round(p81, 4)} \n -----------------');

        # plot LXXXII
        sns.lmplot(x = quant_vars[4], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr82, 3)} | P-value: {round(p82, 4)} \n -----------------');

        # plot LXXXIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr83, 3)} | P-value: {round(p83, 4)} \n -----------------');

        # plot LXXXIV
        sns.lmplot(x = quant_vars[6], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr84, 3)} | P-value: {round(p84, 4)} \n -----------------');

        # plot LXXXV
        sns.lmplot(x = quant_vars[7], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr85, 3)} | P-value: {round(p85, 4)} \n -----------------');

        # plot LXXXVI
        sns.lmplot(x = quant_vars[8], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr86, 3)} | P-value: {round(p86, 4)} \n -----------------');

        # plot LXXXVII
        sns.lmplot(x = quant_vars[9], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr87, 3)} | P-value: {round(p87, 4)} \n -----------------');

        # plot LXXXVIII
        sns.lmplot(x = quant_vars[10], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr88, 3)} | P-value: {round(p88, 4)} \n -----------------');

        # plot LXXXIX
        sns.lmplot(x = quant_vars[11], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr89, 3)} | P-value: {round(p89, 4)} \n -----------------');

        # plot XC
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr90, 3)} | P-value: {round(p90, 4)} \n -----------------');

        # plot XCI
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr91, 3)} | P-value: {round(p91, 4)} \n -----------------');

        # plot XCII
        sns.lmplot(x = quant_vars[0], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr92, 3)} | P-value: {round(p92, 4)} \n -----------------');

        # plot XCIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr93, 3)} | P-value: {round(p93, 4)} \n -----------------');

        # plot XCIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr94, 3)} | P-value: {round(p94, 4)} \n -----------------');

        # plot XCV
        sns.lmplot(x = quant_vars[3], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr95, 3)} | P-value: {round(p95, 4)} \n -----------------');

        # plot XCVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr96, 3)} | P-value: {round(p96, 4)} \n -----------------');

        # plot XCVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr97, 3)} | P-value: {round(p97, 4)} \n -----------------');

        # plot XCVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr98, 3)} | P-value: {round(p98, 4)} \n -----------------');

        # plot XCIX
        sns.lmplot(x = quant_vars[7], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr99, 3)} | P-value: {round(p99, 4)} \n -----------------');

        # plot C
        sns.lmplot(x = quant_vars[8], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr100, 3)} | P-value: {round(p100, 4)} \n -----------------');

        # plot CI
        sns.lmplot(x = quant_vars[9], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr101, 3)} | P-value: {round(p101, 4)} \n -----------------');

        # plot CII
        sns.lmplot(x = quant_vars[10], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr102, 3)} | P-value: {round(p102, 4)} \n -----------------');

        # plot CIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr103, 3)} | P-value: {round(p103, 4)} \n -----------------');

        # plot CIV
        sns.lmplot(x = quant_vars[12], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr104, 3)} | P-value: {round(p104, 4)} \n -----------------');

        # plot CV
        sns.lmplot(x = quant_vars[13], y = quant_vars[14], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr105, 3)} | P-value: {round(p105, 4)} \n -----------------');

        # plot CVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr106, 3)} | P-value: {round(p106, 4)} \n -----------------');

        # plot CVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr107, 3)} | P-value: {round(p107, 4)} \n -----------------');

        # plot CVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr108, 3)} | P-value: {round(p108, 4)} \n -----------------');

        # plot CVIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr109, 3)} | P-value: {round(p109, 4)} \n -----------------');

        # plot CX
        sns.lmplot(x = quant_vars[4], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr110, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr111, 3)} | P-value: {round(p111, 4)} \n -----------------');

        # plot CXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr112, 3)} | P-value: {round(p112, 4)} \n -----------------');

        # plot CXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr113, 3)} | P-value: {round(p113, 4)} \n -----------------');

        # plot CXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr114, 3)} | P-value: {round(p114, 4)} \n -----------------');

        # plot CXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr115, 3)} | P-value: {round(p115, 4)} \n -----------------');

        # plot CXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr116, 3)} | P-value: {round(p116, 4)} \n -----------------');

        # plot CXVII
        sns.lmplot(x = quant_vars[11], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr117, 3)} | P-value: {round(p117, 4)} \n -----------------');

        # plot CXVIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr118, 3)} | P-value: {round(p118, 4)} \n -----------------');

        # plot CXIX
        sns.lmplot(x = quant_vars[13], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr119, 3)} | P-value: {round(p119, 4)} \n -----------------');

        # plot CXX
        sns.lmplot(x = quant_vars[14], y = quant_vars[15], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr120, 3)} | P-value: {round(p110, 4)} \n -----------------');

        # plot CXXI
        sns.lmplot(x = quant_vars[0], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr121, 3)} | P-value: {round(p121, 4)} \n -----------------');

        # plot CXXII
        sns.lmplot(x = quant_vars[1], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr122, 3)} | P-value: {round(p122, 4)} \n -----------------');

        # plot CXXIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr123, 3)} | P-value: {round(p123, 4)} \n -----------------');

        # plot CXXIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr124, 3)} | P-value: {round(p124, 4)} \n -----------------');

        # plot CXXV
        sns.lmplot(x = quant_vars[4], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr125, 3)} | P-value: {round(p125, 4)} \n -----------------');

        # plot CXXVI
        sns.lmplot(x = quant_vars[5], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr126, 3)} | P-value: {round(p126, 4)} \n -----------------');

        # plot CXXVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr127, 3)} | P-value: {round(p127, 4)} \n -----------------');

        # plot CXXVIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr128, 3)} | P-value: {round(p128, 4)} \n -----------------');

        # plot CXXIX
        sns.lmplot(x = quant_vars[8], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr129, 3)} | P-value: {round(p129, 4)} \n -----------------');

        # plot CXXX
        sns.lmplot(x = quant_vars[9], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr130, 3)} | P-value: {round(p130, 4)} \n -----------------');

        # plot CXXXI
        sns.lmplot(x = quant_vars[10], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr131, 3)} | P-value: {round(p131, 4)} \n -----------------');

        # plot CXXXII
        sns.lmplot(x = quant_vars[11], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr132, 3)} | P-value: {round(p132, 4)} \n -----------------');

        # plot CXXXIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr133, 3)} | P-value: {round(p133, 4)} \n -----------------');

        # plot CXXXIV
        sns.lmplot(x = quant_vars[13], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr134, 3)} | P-value: {round(p134, 4)} \n -----------------');

        # plot CXXXV
        sns.lmplot(x = quant_vars[14], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr135, 3)} | P-value: {round(p135, 4)} \n -----------------');

        # plot CXXXVI
        sns.lmplot(x = quant_vars[15], y = quant_vars[16], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr136, 3)} | P-value: {round(p136, 4)} \n -----------------');

        # plot CXXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr137, 3)} | P-value: {round(p137, 4)} \n -----------------');

        # plot CXXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr138, 3)} | P-value: {round(p138, 4)} \n -----------------');

        # plot CXXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr139, 3)} | P-value: {round(p139, 4)} \n -----------------');

        # plot CXL
        sns.lmplot(x = quant_vars[3], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr140, 3)} | P-value: {round(p140, 4)} \n -----------------');

        # plot CXLI
        sns.lmplot(x = quant_vars[4], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr141, 3)} | P-value: {round(p141, 4)} \n -----------------');

        # plot CXLII
        sns.lmplot(x = quant_vars[5], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr142, 3)} | P-value: {round(p142, 4)} \n -----------------');

        # plot CXLIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr143, 3)} | P-value: {round(p143, 4)} \n -----------------');

        # plot CXLIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr144, 3)} | P-value: {round(p144, 4)} \n -----------------');

        # plot CXLV
        sns.lmplot(x = quant_vars[8], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr145, 3)} | P-value: {round(p145, 4)} \n -----------------');

        # plot CXLVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr146, 3)} | P-value: {round(p146, 4)} \n -----------------');

        # plot CXLVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr147, 3)} | P-value: {round(p147, 4)} \n -----------------');

        # plot CXLVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr148, 3)} | P-value: {round(p148, 4)} \n -----------------');

        # plot CXLIX
        sns.lmplot(x = quant_vars[12], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr149, 3)} | P-value: {round(p149, 4)} \n -----------------');

        # plot CL
        sns.lmplot(x = quant_vars[13], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr150, 3)} | P-value: {round(p150, 4)} \n -----------------');

        # plot CLI
        sns.lmplot(x = quant_vars[14], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr151, 3)} | P-value: {round(p151, 4)} \n -----------------');

        # plot CLII
        sns.lmplot(x = quant_vars[15], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr152, 3)} | P-value: {round(p152, 4)} \n -----------------');

        # plot CLIII
        sns.lmplot(x = quant_vars[16], y = quant_vars[17], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr153, 3)} | P-value: {round(p153, 4)} \n -----------------');

        # plot CLIV
        sns.lmplot(x = quant_vars[0], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr154, 3)} | P-value: {round(p154, 4)} \n -----------------');

        # plot CLV
        sns.lmplot(x = quant_vars[1], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr155, 3)} | P-value: {round(p155, 4)} \n -----------------');

        # plot CLVI
        sns.lmplot(x = quant_vars[2], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr156, 3)} | P-value: {round(p156, 4)} \n -----------------');

        # plot CLVII
        sns.lmplot(x = quant_vars[3], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr157, 3)} | P-value: {round(p157, 4)} \n -----------------');

        # plot CLVIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr158, 3)} | P-value: {round(p158, 4)} \n -----------------');

        # plot CLIX
        sns.lmplot(x = quant_vars[5], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr159, 3)} | P-value: {round(p159, 4)} \n -----------------');

        # plot CLX
        sns.lmplot(x = quant_vars[6], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr160, 3)} | P-value: {round(p160, 4)} \n -----------------');

        # plot CLXI
        sns.lmplot(x = quant_vars[7], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr161, 3)} | P-value: {round(p161, 4)} \n -----------------');

        # plot CLXII
        sns.lmplot(x = quant_vars[8], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr162, 3)} | P-value: {round(p162, 4)} \n -----------------');

        # plot CLXIII
        sns.lmplot(x = quant_vars[9], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr163, 3)} | P-value: {round(p163, 4)} \n -----------------');

        # plot CLXIV
        sns.lmplot(x = quant_vars[10], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr164, 3)} | P-value: {round(p164, 4)} \n -----------------');

        # plot CLXV
        sns.lmplot(x = quant_vars[11], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr165, 3)} | P-value: {round(p165, 4)} \n -----------------');

        # plot CLXVI
        sns.lmplot(x = quant_vars[12], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr166, 3)} | P-value: {round(p166, 4)} \n -----------------');

        # plot CLXVII
        sns.lmplot(x = quant_vars[13], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr167, 3)} | P-value: {round(p167, 4)} \n -----------------');

        # plot CLXVIII
        sns.lmplot(x = quant_vars[14], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr168, 3)} | P-value: {round(p168, 4)} \n -----------------');

        # plot CLXIX
        sns.lmplot(x = quant_vars[15], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr169, 3)} | P-value: {round(p169, 4)} \n -----------------');

        # plot CLXX
        sns.lmplot(x = quant_vars[16], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr170, 3)} | P-value: {round(p170, 4)} \n -----------------');

        # plot CLXXI
        sns.lmplot(x = quant_vars[17], y = quant_vars[18], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr171, 3)} | P-value: {round(p171, 4)} \n -----------------');

        # plot CLXXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr172, 3)} | P-value: {round(p172, 4)} \n -----------------');

        # plot CLXXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr173, 3)} | P-value: {round(p173, 4)} \n -----------------');

        # plot CLXXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr174, 3)} | P-value: {round(p174, 4)} \n -----------------');

        # plot CLXXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr175, 3)} | P-value: {round(p175, 4)} \n -----------------');

        # plot CLXXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr176, 3)} | P-value: {round(p176, 4)} \n -----------------');

        # plot CLXXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr177, 3)} | P-value: {round(p177, 4)} \n -----------------');

        # plot CLXXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr178, 3)} | P-value: {round(p178, 4)} \n -----------------');

        # plot CLXXIX
        sns.lmplot(x = quant_vars[7], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr179, 3)} | P-value: {round(p179, 4)} \n -----------------');

        # plot CLXXX
        sns.lmplot(x = quant_vars[8], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr180, 3)} | P-value: {round(p180, 4)} \n -----------------');

        # plot CLXXXI
        sns.lmplot(x = quant_vars[9], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr181, 3)} | P-value: {round(p181, 4)} \n -----------------');

        # plot CLXXXII
        sns.lmplot(x = quant_vars[10], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr182, 3)} | P-value: {round(p182, 4)} \n -----------------');

        # plot CLXXXIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr183, 3)} | P-value: {round(p183, 4)} \n -----------------');

        # plot CLXXXIV
        sns.lmplot(x = quant_vars[12], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr184, 3)} | P-value: {round(p184, 4)} \n -----------------');

        # plot CLXXXV
        sns.lmplot(x = quant_vars[13], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr185, 3)} | P-value: {round(p185, 4)} \n -----------------');

        # plot CLXXXVI
        sns.lmplot(x = quant_vars[14], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr186, 3)} | P-value: {round(p186, 4)} \n -----------------');

        # plot CLXXXVII
        sns.lmplot(x = quant_vars[15], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr187, 3)} | P-value: {round(p187, 4)} \n -----------------');

        # plot CLXXXVIII
        sns.lmplot(x = quant_vars[16], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr188, 3)} | P-value: {round(p188, 4)} \n -----------------');

        # plot CLXXXIX
        sns.lmplot(x = quant_vars[17], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr189, 3)} | P-value: {round(p189, 4)} \n -----------------');

        # plot CXC
        sns.lmplot(x = quant_vars[18], y = quant_vars[19], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr190, 3)} | P-value: {round(p190, 4)} \n -----------------');

        # plot CXCI
        sns.lmplot(x = quant_vars[0], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr191, 3)} | P-value: {round(p191, 4)} \n -----------------');

        # plot CXCII
        sns.lmplot(x = quant_vars[1], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr192, 3)} | P-value: {round(p192, 4)} \n -----------------');

        # plot CXCIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr193, 3)} | P-value: {round(p193, 4)} \n -----------------');

        # plot CXCIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr194, 3)} | P-value: {round(p194, 4)} \n -----------------');

        # plot CXCV
        sns.lmplot(x = quant_vars[4], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr195, 3)} | P-value: {round(p195, 4)} \n -----------------');

        # plot CXCVI
        sns.lmplot(x = quant_vars[5], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr196, 3)} | P-value: {round(p196, 4)} \n -----------------');

        # plot CXCVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr197, 3)} | P-value: {round(p197, 4)} \n -----------------');

        # plot CXCVIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr198, 3)} | P-value: {round(p198, 4)} \n -----------------');

        # plot CXCIX
        sns.lmplot(x = quant_vars[8], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr199, 3)} | P-value: {round(p199, 4)} \n -----------------');

        # plot CC
        sns.lmplot(x = quant_vars[9], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr200, 3)} | P-value: {round(p200, 4)} \n -----------------');

        # plot CCI
        sns.lmplot(x = quant_vars[10], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr201, 3)} | P-value: {round(p201, 4)} \n -----------------');

        # plot CCII
        sns.lmplot(x = quant_vars[11], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr202, 3)} | P-value: {round(p202, 4)} \n -----------------');

        # plot CCIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr203, 3)} | P-value: {round(p203, 4)} \n -----------------');

        # plot CCIV
        sns.lmplot(x = quant_vars[13], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr204, 3)} | P-value: {round(p204, 4)} \n -----------------');

        # plot CCV
        sns.lmplot(x = quant_vars[14], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr205, 3)} | P-value: {round(p205, 4)} \n -----------------');

        # plot CCVI
        sns.lmplot(x = quant_vars[15], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr206, 3)} | P-value: {round(p206, 4)} \n -----------------');

        # plot CCVII
        sns.lmplot(x = quant_vars[16], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr207, 3)} | P-value: {round(p207, 4)} \n -----------------');

        # plot CCVIII
        sns.lmplot(x = quant_vars[17], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr208, 3)} | P-value: {round(p208, 4)} \n -----------------');

        # plot CCIX
        sns.lmplot(x = quant_vars[18], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr209, 3)} | P-value: {round(p209, 4)} \n -----------------');

        # plot CCX
        sns.lmplot(x = quant_vars[19], y = quant_vars[20], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr210, 3)} | P-value: {round(p210, 4)} \n -----------------');

        # plot CCXI
        sns.lmplot(x = quant_vars[0], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr211, 3)} | P-value: {round(p211, 4)} \n -----------------');

        # plot CCXII
        sns.lmplot(x = quant_vars[1], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr212, 3)} | P-value: {round(p212, 4)} \n -----------------');

        # plot CCXII
        sns.lmplot(x = quant_vars[2], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr213, 3)} | P-value: {round(p213, 4)} \n -----------------');

        # plot CCXIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr214, 3)} | P-value: {round(p214, 4)} \n -----------------');

        # plot CCXV
        sns.lmplot(x = quant_vars[4], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr215, 3)} | P-value: {round(p215, 4)} \n -----------------');

        # plot CCXVI
        sns.lmplot(x = quant_vars[5], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr216, 3)} | P-value: {round(p216, 4)} \n -----------------');

        # plot CCXVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr217, 3)} | P-value: {round(p217, 4)} \n -----------------');

        # plot CCXVIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr218, 3)} | P-value: {round(p218, 4)} \n -----------------');

        # plot CCXIX
        sns.lmplot(x = quant_vars[8], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr219, 3)} | P-value: {round(p219, 4)} \n -----------------');

        # plot CCXX
        sns.lmplot(x = quant_vars[9], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr220, 3)} | P-value: {round(p220, 4)} \n -----------------');

        # plot CCXXI
        sns.lmplot(x = quant_vars[10], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr221, 3)} | P-value: {round(p221, 4)} \n -----------------');

        # plot CCXXII
        sns.lmplot(x = quant_vars[11], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr222, 3)} | P-value: {round(p222, 4)} \n -----------------');

        # plot CCXXIII
        sns.lmplot(x = quant_vars[12], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr223, 3)} | P-value: {round(p223, 4)} \n -----------------');

        # plot CCXXIV
        sns.lmplot(x = quant_vars[13], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr224, 3)} | P-value: {round(p224, 4)} \n -----------------');

        # plot CCXXV
        sns.lmplot(x = quant_vars[14], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr225, 3)} | P-value: {round(p225, 4)} \n -----------------');

        # plot CCXXVI
        sns.lmplot(x = quant_vars[15], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr226, 3)} | P-value: {round(p226, 4)} \n -----------------');

        # plot CCXXVII
        sns.lmplot(x = quant_vars[16], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr227, 3)} | P-value: {round(p227, 4)} \n -----------------');

        # plot CCXXVIII
        sns.lmplot(x = quant_vars[17], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr228, 3)} | P-value: {round(p228, 4)} \n -----------------');

        # plot CCXXIX
        sns.lmplot(x = quant_vars[18], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr229, 3)} | P-value: {round(p229, 4)} \n -----------------');

        # plot CCXXX
        sns.lmplot(x = quant_vars[19], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr230, 3)} | P-value: {round(p230, 4)} \n -----------------');

        # plot CCXXXI
        sns.lmplot(x = quant_vars[20], y = quant_vars[21], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr231, 3)} | P-value: {round(p231, 4)} \n -----------------');


        


# <^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^> CUSTOMIZED RETURNS ON STATS TESTS FOUND HERE <^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^># 




# return_chi2 defines one parameter, an observed cross-tabulation, runs the stats.chi2_contingency function and returns the test results in a readable format.
def return_chi2(observed):
    
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





# # # /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ FEATURE SELECTION FUNCTIONS HERE /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ # # # 



# # # Note: must define X_train and y_train prior to running feature selection functions
# # note: also these output lists are ordered backward

# X_train = predictors or features (same thing if you got the right features)
# y_train = target
# k = number of features you want

# select_kbest defines 3 parameters, X_train (predictors), y_train (target variable) and k (number of features to spit), and returns a list of the best features my man
def select_kbest(X_train, y_train, k):

    # import feature selection tools
    from sklearn.feature_selection import SelectKBest, f_regression

    # create the selector
    f_select = SelectKBest(f_regression, k = k)

    # fit the selector
    f_select.fit(X_train, y_train)

    # create a boolean mask to show if feature was selected
    feat_mask = f_select.get_support()
    
    # create a list of the best features
    best_features = X_train.iloc[:,feat_mask].columns.to_list()

    # gimme gimme
    return best_features



# rfe defines 3 parameters, X_train (features), y_train (target variable) and k (number of features to bop), and returns a list of the best boppits m8
def rfe(X_train, y_train, k):

    # import feature selection tools
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression

    # crank it
    lm = LinearRegression()

    # pop it
    rfe = RFE(lm, k)
    
    # bop it
    rfe.fit(X_train, y_train)  
    
    # twist it
    feat_mask = rfe.support_
    
    # pull it 
    best_rfe = X_train.iloc[:,feat_mask].columns.tolist()
    
    # bop it
    return best_rfe
