import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
import eyal_funcs as EB



if __name__ == '__main__':
    df = EB.load_data(folder='14-K-Nearest-Neighbors/', file_name='KNN_Project_Data')
    df.head()

    ## explore data
    sns.pairplot(data = df, hue = 'TARGET CLASS')

    ## Standardize Variables
    std_scaler = StandardScaler()
    # fit to predict TARGET CLASS
    std_scaler.fit(df.drop('TARGET CLASS', axis=1))
    std_scaled_features = std_scaler.transform(df.drop('TARGET CLASS', axis=1))

    # sneek-peek for verification of scaling
    df_feat = pd.DataFrame(std_scaled_features, columns=df.columns[:-1]) #create DataFrame off features
    df_feat.head()

    # Train-test-split and train
    X_train, X_test, y_train, y_test = train_test_split(std_scaled_features, df['TARGET CLASS'], test_size=0.30, )
    knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train) # create insatnce for knn

    # Predictions and evaluations:
    pred = knn.predict(X_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    # colustion : precision is roughly ok, but still low, should increase K and exmplore optimal K

    # Chossing optinal k
    error_rate = [] # drawing the error for each k

    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
    # vizulize optimal-k results
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')

    # chosen k to be = 25
    knn = KNeighborsClassifier(n_neighbors=25)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print('WITH K=25')
    print('\n')
    print(confusion_matrix(y_test, pred))
    print('\n')
    print(classification_report(y_test, pred))

    # imporvment of 15% in precision and f1-score





