import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import stats

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr


sns.set()

#plot_variable_pairs defines two parameters, a dataframe and a list of numeric colummns up for grabs, and returns several lmplots and correlation coefficients as well as a reference p-value.
def plot_variable_pairs(df, quant_vars):

    #determine correlation coefficients
    corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
    corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
    corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])

    #plot relationships between continuous variables (3 pairs or 3 relationships here, and 3 continuous features)
    sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');
    sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');
    sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');


#plot_categorical_and_continuous defines 3 parameters, a dataframe to pull data from, and x variable (categorical column) and a y variable (continuous value column), and returns visualizations of these relationships.
def plot_categorical_and_continuous(df, x, y):

    #plot 3 figures and 3 different plots for visualizing categorical-continuous relationships
    plt.figure(figsize = (8, 5))
    sns.boxplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.stripplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.violinplot(x = x, y = y, data = df, palette = 'inferno_r');

#plot 6 pairs (4 continuous features)
def plot_vp_ext(df, quant_vars):
    
    #determine correlation coefficients
    corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
    corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
    corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
    corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
    corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
    corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])

    #plot relationships between continuous variables
    sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr1, 3)} | P-value: {p1} \n -----------------');
    sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr2, 3)} | P-value: {p2} \n -----------------');
    sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr3, 3)} | P-value: {p3} \n -----------------');
    sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr4, 3)} | P-value: {p4} \n -----------------');
    sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr5, 3)} | P-value: {p5} \n -----------------');
    sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr6, 3)} | P-value: {p6} \n -----------------');

#plot 1 pair (2 continuous features)
def plot_variable_pair(df, quant_vars):

    #determine correlation coefficient
    corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])

    #plot relationships between continuous variables
    sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

#plot 10 pairs (5 features)
def plot_vp_adv(df, quant_vars):

    #determine correlation coefficients
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

    #plot relationships between continuous variables
    #plot I
    sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

    #plot II
    sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

    #plot III
    sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

    #plot IV
    sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

    #plot V
    sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr5, 3)} | P-value: {round*p5, 4)} \n -----------------');

    #plot VI
    sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

    #plot VII
    sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

    #plot VIII
    sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

    #plot IX
    sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');
    
    #plot X
    sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

#plot 15 pairs or 6 features
def plot_15_vp(df, quant_vars):

    #determine correlation coefficients
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

    #plot relationships between continuous variables
    #plot I
    sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

    #plot II
    sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

    #plot III
    sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

    #plot IV
    sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

    #plot V
    sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr5, 3)} | P-value: {round*p5, 4)} \n -----------------');

    #plot VI
    sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

    #plot VII
    sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

    #plot VIII
    sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

    #plot IX
    sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');
    
    #plot X
    sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');)

    #plot XI
    sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

    #plot XII
    sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

    #plot XIII
    sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

    #plot XIV
    sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');
    
    #plot XV
    sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');)


#plot 21 pairs or 7 features
def plot_21_vp(df, quant_vars):

    #determine correlation coefficients
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
    


    #plot relationships between continuous variables
    #plot I
    sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

    #plot II
    sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

    #plot III
    sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

    #plot IV
    sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

    #plot V
    sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr5, 3)} | P-value: {round*p5, 4)} \n -----------------');

    #plot VI
    sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

    #plot VII
    sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

    #plot VIII
    sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

    #plot IX
    sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');
    
    #plot X
    sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');)

    #plot XI
    sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

    #plot XII
    sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

    #plot XIII
    sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

    #plot XIV
    sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');
    
    #plot XV
    sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');)

    #plot XVI
    sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');)

    #plot XVII
    sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

    #plot XVIII
    sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------');

    #plot XIX
    sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------');

    #plot XX
    sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------');
    
    #plot XXI
    sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');)


#plot 28 pairs or 8 features
def plot_28_vp(df, quant_vars):

    #determine correlation coefficients
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
    
    


    #plot relationships between continuous variables
    #plot I
    sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

    #plot II
    sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

    #plot III
    sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

    #plot IV
    sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

    #plot V
    sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr5, 3)} | P-value: {round*p5, 4)} \n -----------------');

    #plot VI
    sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

    #plot VII
    sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

    #plot VIII
    sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

    #plot IX
    sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');
    
    #plot X
    sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');)

    #plot XI
    sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

    #plot XII
    sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

    #plot XIII
    sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

    #plot XIV
    sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');
    
    #plot XV
    sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');)

    #plot XVI
    sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');)

    #plot XVII
    sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

    #plot XVIII
    sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------');

    #plot XIX
    sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------');

    #plot XX
    sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------');
    
    #plot XXI
    sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');)

    #plot XXII
    sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');
    
    #plot XXIII
    sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');)

    #plot XXIV
    sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');)

    #plot XXV
    sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

    #plot XXVI
    sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------');

    #plot XXVII
    sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------');

    #plot XXVIII
    sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------');
    
    #plot XXIV
    sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');)