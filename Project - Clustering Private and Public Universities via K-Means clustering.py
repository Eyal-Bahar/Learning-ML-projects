import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report,confusion_matrix

import eyal_funcs as EB

if __name__ == '__main__':

    Note = 'We are pretending that we dont know the private\public labeling'


    df = EB.load_data(folder='17-K-Means-Clustering/', file_name='College_Data')
    # Explore dataframe
    df.head()
    df.info()
    df.describe()

    # Vizualize
    plt.figure(1)
    sns.scatterplot(y='Grad.Rate', x='Room.Board', data=df, hue='Private')

    plt.figure(2)
    sns.scatterplot(y='F.Undergrad', x='Outstate', data=df, hue='Private')

    plt.figure(3)
    df[df['Private'] == 'Yes']['Outstate'].hist()
    df[df['Private'] == 'No']['Outstate'].hist()

    plt.figure(4)
    df[df['Private'] == 'Yes']['Grad.Rate'].hist(bins = 20)
    df[df['Private'] == 'No']['Grad.Rate'].hist(bins = 20)

    df[df['Grad.Rate']>100] = 100  # cleaning data (cant have more than 100% grad rate!)

    # fit to a 2 cluster model

    kmeans = KMeans(n_clusters=2)
    df.set_index('Unnamed: 0', inplace = True)
    kmeans.fit(df.drop('Private', axis=1))

    # The cluster center vectors
    kmeans.cluster_centers_

    # Evaluation
    # Since we have the labels... lets test it out
    def true_to_one(x):
        if x=='Yes':
            return 1
        else:
            return 0

    df['Cluster'] = df['Private'].apply(true_to_one)
    confusion_matrix(df['Cluster'], kmeans.labels_)

    print(classification_report(df['Cluster'], kmeans.labels_))









