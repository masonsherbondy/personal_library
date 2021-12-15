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

    #plot relationships between continuous variables
    sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr1, 3)} | P-value: {p1} \n -----------------');
    sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr2, 3)} | P-value: {p2} \n -----------------');
    sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr3, 3)} | P-value: {p3} \n -----------------');


#plot_categorical_and_continuous defines 3 parameters, a dataframe to pull data from, and x variable (categorical column) and a y variable (continuous value column), and returns visualizations of these relationships.
def plot_categorical_and_continuous(df, x, y):

    #plot 3 figures and 3 different plots for visualizing categorical-continuous relationships
    plt.figure(figsize = (8, 5))
    sns.boxplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.stripplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.violinplot(x = x, y = y, data = df, palette = 'inferno_r');


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