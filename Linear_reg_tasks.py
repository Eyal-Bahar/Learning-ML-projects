import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Press the green button in the gutter to run the script.
def Linear_regression_lesson:
    path_TAU = "D:/Dropbox/Eyal Bahar/Phd - Tel Aviv/פייתון/udemy/python_for_DS_ML/Py_DS_ML_Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/"
    file_name = "USA_Housing.csv"
    USAhousing = pd.read_csv(path_TAU + file_name)
    sns.pairplot(USAhousing)

    ## Doing a regression model
    from sklearn.model_selection import train_test_split
    X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                    'Avg. Area Number of Bedrooms', 'Area Population']]
    y = USAhousing['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
    ##
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    ## Evaluate our model by coeeficiencs
    print(lm.intercept_)
    cdf = pd.DataFrame(lm.coef_, X.columns, columns=["Coeff"])  # CoefficiectDataFrat

    ## pridictions:
    predictions = lm.predict(X_test)
    print(predictions)
    plt.figure(2)
    plt.scatter(y_test, predictions)
    plt.figure(3)
    sns.displot(y_test - predictions)

    ## Regression evlaution metrics
    from sklearn import metrics
    metrics.mean_squared_error(y_test, predictions)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test, predictions))

def Linear_Regression_project():
    # Linear Regression Project Congratulations! You just got some contract work with an Ecommerce company
    # based in New York City that sells clothing online but they also have in-store style and clothing advice
    # sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can
    # go home and order either on a mobile app or website for the clothes they want. The company is trying to
    # decide whether to focus their efforts on their mobile app experience or their website.

    # They've hired you on contract to help them figure it out! Let's get started!
    # Just follow the steps below to analyze the customer data (it's fake, don't worry I didn't give you real
    # credit card numbers or emails).

    # Read Ecommerce Customers csv file as a DataFrame
    path_TAU = "D:/Dropbox/Eyal Bahar/Phd - Tel Aviv/פייתון/udemy/python_for_DS_ML/Py_DS_ML_Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/"
    file_name = "Ecommerce Customers"
    customers = pd.read_csv(path_TAU + file_name)
    ##
    customers.info()
    customers.describe()
    ##
    plt.figure(1)
    sns.jointplot(x=customers['Time on Website'], y=customers['Yearly Amount Spent'])
    ##
    plt.figure(2)
    sns.jointplot(x=customers['Time on App'], y=customers['Yearly Amount Spent'])
    ##
    plt.figure(3)
    sns.jointplot(x=customers['Time on App'], y=customers['Length of Membership'], kind="hex")
    ##
    plt.figure(4)
    sns.pairplot(customers, height=2.5, diag_kind = "hist", diag_kws = {'alpha':0.55, 'bins':10})
    ## Based off this plot the most correlated feature with Yearly Amount Spent is length of membership
    plt.figure(5)
    sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data = customers)

    ## Training and Testing Data
    # split the data into training and testing sets.
    # Set a variable X equal to the numerical features of the customers and
    # a variable y equal to the "Yearly Amount Spent" column
    X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y = customers['Yearly Amount Spent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Training
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    LR.coef_
    # predict
    LR.predictions = LR.predict(X_test)
    plt.scatter(x = y_test,y = LR.predict(X_test))
    plt.xlabel('y_test')
    plt.ylabel('X_test')

    # Mean Absolute Error
    print('MAE:', metrics.mean_absolute_error(y_test, LR.predictions))
    # Mean Squared Error
    print('MSE:', metrics.mean_squared_error(y_test, LR.predictions))
    # Root Mean Squared Error
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, LR.predictions)))


if __name__ == '__main__':
    Linear_Regression_project()