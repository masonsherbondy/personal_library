# This is the way
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

## DEPTH CHARGE FUNCTIONS ##

# Decision Tree or Random Forest
# K-Nearest Neighbor
# TBC...



# Decision Tree or Random Forest 
def tree_train_validate(tree_model,
                        X_train,
                        y_train, 
                        X_validate, 
                        y_validate, 
                        leaves = 11,
                        criterion = 'gini', 
                        min_samples_leaf = 3, 
                        random_state = 421
                        ):

    '''
    This function accepts a decision tree or random forest classifier model (1 argument), train and validate modeling sets (4 arguments), the amount of leaves
    or depth you want (1 argument, default 11), model criterion (default = gini), a specified minimum number of samples per leaf (default = 3)
    and a random state number (default = 421), and it prints out train and validate accuracy results for specified (first argument) different depths and 
    plots the validate accuracy results for different depths.
    '''

    depth_range = range(1, leaves + 1)    # set a range of depths to explore
    train_scores = []    # set an empty list for train scores
    validate_scores = []    # set an empty list for validate scores
    metrics = []    # set an empty list for dictionaries

    for depth in depth_range:    # commence loop through different max depths for Decision Tree
        model = tree_model(max_depth = depth, criterion = criterion, min_samples_leaf = min_samples_leaf, random_state = random_state)    # create object
        model.fit(X_train, y_train)    # fit object
        train_scores.append(model.score(X_train, y_train))    # add train scores to list
        validate_scores.append(model.score(X_validate, y_validate))    # add validate scores to list
        in_sample_accuracy = model.score(X_train, y_train)    # calculate accuracy on train set
        out_of_sample_accuracy = model.score(X_validate, y_validate)    # calculate accuracy on validate set
        output = {                                       # create dictionary with max_depth, train set accuracy and validate accuracy
            'max_depth': depth,                          
            'train_accuracy': in_sample_accuracy,        
            'validate_accuracy': out_of_sample_accuracy
        }

        metrics.append(output)    # add dictionaries to list

    metrics_df = pd.DataFrame(metrics)    # form dataframe from scores data
    metrics_df = metrics_df.set_index('max_depth')    # set index to max depth
    metrics_df['difference'] = metrics_df.train_accuracy - metrics_df.validate_accuracy    # create column of values for difference between train and validate
    metrics_df['absolute'] = metrics_df['difference'].apply(lambda x: abs(x))    # create column of absolute distance between train and validate accuracies
    differences = metrics_df.absolute.to_list()    # generate list of distances between accuracy scores

    plt.figure(figsize = (11, 7))    # create figure
    plt.xlabel('Depth (Leaves)')    # label x-axis
    plt.ylabel('Accuracy')    # label y-axis

    if tree_model == DecisionTreeClassifier:
        plt.title('Decision Tree Accuracies')    # title
        plt.scatter(depth_range, train_scores, color = 'darkorange', label = 'Train Accuracy')    # plot scatter between depth range and train accuracy
        plt.plot(depth_range, train_scores, color = 'gold')    # plot line between depth range and train accuracy

    else:
        plt.title(f'Random Forest Accuracies')    # title
        plt.scatter(depth_range, train_scores, color = 'skyblue', label = 'Train Accuracy')    # plot scatter between depth range and train accuracy
        plt.plot(depth_range, train_scores, color = 'navy')    # plot line between depth range and train accuracy

    plt.scatter(depth_range, validate_scores, color = 'forestgreen', label = 'Validate Accuracy')    # plot scatter between depth range and validate accuracy
    plt.plot(depth_range, validate_scores, color = 'saddlebrown')    # plot line between depth range and validate accuracy
    plt.scatter(depth_range, differences, color = 'darkslategrey', label = 'Difference')    # plot difference
    plt.plot(depth_range, differences, color = 'black')    # plot difference

    plt.grid(True)
    plt.legend()    # include legend for clarity
    plt.show();
                  
    print(metrics_df)     # view metrics dataframe

# K-Nearest Neighbor
def KNN_metrics(X_train, y_train, X_validate, y_validate, weights = 'uniform', depth = 19):

    '''
    Docstring go here
    '''

    k_range = range(1, depth + 1)
    training_scores = []
    validate_scores = []
    metrics = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k, weights = weights)
        knn.fit(X_train, y_train)
        training_scores.append(knn.score(X_train, y_train))
        validate_scores.append(knn.score(X_validate, y_validate))
        in_sample_accuracy = knn.score(X_train, y_train)
        out_of_sample_accuracy = knn.score(X_validate, y_validate)
        output = {
            'n_neighbors': k,
            'train_accuracy': in_sample_accuracy,
            'validate_accuracy': out_of_sample_accuracy
        }

        metrics.append(output)
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.set_index('n_neighbors')
    metrics_df['difference'] = metrics_df.train_accuracy - metrics_df.validate_accuracy
    metrics_df['absolute'] = metrics_df.difference.apply(lambda x: abs(x))
    differences = metrics_df.absolute.to_list()

    plt.figure(figsize = (11, 7))
    plt.xlabel('Depth (K)')
    plt.ylabel('Accuracy')
    plt.scatter(k_range, training_scores, color = 'crimson', label = 'Training Accuracy')
    plt.plot(k_range, training_scores, color = 'maroon')
    plt.scatter(k_range, validate_scores, color = 'black', label = 'Validate Accuracy')
    plt.plot(k_range, validate_scores, color = 'black')
    plt.scatter(k_range, differences, color = 'royalblue', label = 'Difference')
    plt.plot(k_range, differences, color = 'navy')
    plt.title('KNN Model Accuracies')
    plt.show();
    
    print(metrics_df)