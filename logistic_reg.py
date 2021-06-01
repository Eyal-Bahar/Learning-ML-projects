import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics
import eyal_funcs as EB


if __name__ == '__main__':
    udemy_path = EB.basic_path('PC')
    folder = '13-Logistic-Regression/'
    file_name = 'titanic_train.csv'
    full_path = udemy_path + folder + file_name
    train = pd.read_csv(full_path)

    ## check data completeness
    plt.figure(1)
    sns.heatmap(train.isnull(), yticklabels='false',cbar = 'False', cmap = 'viridis')
    ##
    sns.set_style('whitegrid')
    plt.figure(2)
    sns.countplot(x='Survived', hue='Pclass', data=train)
    ##



