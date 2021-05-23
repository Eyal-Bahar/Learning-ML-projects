import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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
    cdf = pd.DataFrame(lm.coef_, X.columns, columns=["Coeff"]) # CoefficiectDataFrat

    ## pridictions:
    predictions = lm.predict(X_test)
    print(predictions)
    plt.figure(2)
    plt.scatter(y_test, predictions)
    plt.figure(3)
    sns.displot(y_test-predictions)

    ## Regression evlaution metrics
    from sklearn import metrics
    metrics.mean_squared_error(y_test, predictions)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    