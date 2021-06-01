import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
from sklearn.metrics import classification_report
import eyal_funcs as EB

def load_data():
    ## data path
    data_folder = EB.basic_path('PC')
    folder = '13-Logistic-Regression/'
    file_name = 'advertising.csv'
    full_path = data_folder + folder + file_name

    ## load data
    return pd.read_csv(full_path)  # data
    ## General structure of data_set:
    # 'Daily Time Spent on Site': consumer time on site in minutes
    # 'Age': cutomer age in years
    # 'Area Income': Avg. Income of geographical area of consumer
    # 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
    # 'Ad Topic Line': Headline of the advertisement
    # 'City': City of consumer
    # 'Male': Whether or not consumer was male
    # 'Country': Country of consumer
    # 'Timestamp': Time at which consumer clicked on Ad or closed window
    # 'Clicked on Ad': 0 or 1 indicated clicking on Ad

if __name__ == '__main__':
    ## readme: The aim of this project is to predict whether or
    # not a user will click on an ad based off the features of that user.
    add_data = load_data()

    ## Explore data
    add_data.info() # disply columb names and size of dataset
    add_data.describe() # basic statistics of data

    plt.figure(1)
    sns.histplot(add_data.Age,bins=30)
    plt.title('Age hist')

    plt.figure(2)
    sns.jointplot(x='Age', y='Area Income', data=add_data) # Scatter plot showing Area Income versus Age
    plt.title('Avg income in user area vs Age')

    plt.figure(3)
    sns.jointplot(x='Age', y='Daily Time Spent on Site', data=add_data, kind="kde",  color='red' ) # kde distributions
    plt.title('Daily time in site vs Age')

    plt.figure(4)
    sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=add_data)  # scatter distributions
    plt.title('Site usage  vs  Internet Usage (Daily)')

    # plotting multiple variations to emphasize possible data correlations
    plt.figure(5)
    sns.pairplot(data=add_data, hue='Clicked on Ad', height=1,aspect=2)

    ## Logistic Regression
    data_input = ['Area Income']
    measured = 'Clicked on Ad'
    X = add_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
    y = add_data[measured]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    Log_R1 = LogisticRegression()
    Log_R1.fit(X_train, y_train)

    ## Predictions and Evaluations
    predictions = Log_R1.predict(X_test)
    print(classification_report(y_test, predictions))


