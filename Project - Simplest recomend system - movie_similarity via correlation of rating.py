import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report,confusion_matrix
import eyal_funcs as EB


def movie_similarity_Simple_recomendation_system():
    # basic recommendation systems
    # load a subset of movie-lens dataset from file
    column_names = ['user_id', 'item_id', 'rating', 'timestamp'] # i know the names apriori
    df = EB.load_data(folder='19-Recommender-Systems/', file_name='u.data',  sep='\t', names=column_names) # df = pd.read_csv('u.data',)
    movie_titles = EB.load_data(folder='19-Recommender-Systems/', file_name='Movie_Id_Titles')



    # Explore dataframe
    df.head()
    movie_titles.head()
    # movie_titles.info()
    # movie_titles.describe()
    # Merge the movie titles and item id's ( why the heck does it come with the name?!)
    df = pd.merge(df, movie_titles, on='item_id')

    # Explore the data
    df.groupby('title')['rating'].mean().sort_values(ascending=False).head()  # get mean rating
    df.groupby('title')['rating'].count().sort_values(ascending=False).head() # check how many people voted

    ratings = pd.DataFrame(df.groupby('title')['rating'].mean())  # create DataFrame for ratings
    ratings['num of ratings']= pd.DataFrame(df.groupby('title')['rating'].count())

    # Vizualize
    plt.figure(1)
    ratings['num of ratings'].hist(bins=70)
    plt.title('hist of num of ratings')

    plt.figure(2)
    ratings['rating'].hist(bins=70)
    plt.title('hist of rating')

    plt.figure(3)
    sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)

    # Now that we have a general idea of what the data looks like,
    # let's move on to creating a simple recommendation system

    ## Recommending Similar Movies ##
    moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
    moviemat.head()  # alot of NaNs... # Notice
    ratings.sort_values('num of ratings', ascending=False).head(10)
    starwars_user_ratings = moviemat['Star Wars (1977)']
    liarliar_user_ratings = moviemat['Liar Liar (1997)']

    starwars_user_ratings.head()

    ## Get correlations between movies ( assuming that correlated rating distrebutioin is a good score for recommendation)
    similar_to_starwars = moviemat.corrwith(starwars_user_ratings) # correlated the dataset with the starwars user ratings
    similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)  # correlated the dataset with the starwars user ratings
    corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
    ## Filtering irrelavent data:
    # Remove annoying NaN!
    corr_starwars.dropna(inplace=True)
    # remove low count ratings  - based on  hist from figure 2
    corr_starwars = corr_starwars.join(ratings['num of ratings'])
    corr_starwars[corr_starwars['num of ratings'] > 100].sort_values('Correlation',ascending=False).head(10)

    # I could do the same for liarliar but you got the point..


if __name__ == '__main__':
    movie_similarity_Simple_recomendation_system()