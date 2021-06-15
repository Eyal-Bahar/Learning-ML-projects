import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import eyal_funcs as EB
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    """
    This project helps with determining the rating (1 to 5) of a 
    YELP review based on the input text.
    
    The "cool" column is the number of "cool" votes this review received from other Yelp users (limitless). 
    The "useful" and "funny" columns are similar to the "cool" column.

    The Data is from a Kaggle Data Set @ https://www.kaggle.com/c/yelp-recsys-2013
        
    """

    df = pd.read_csv('yelp.csv')

    ## Exploring data
    df.head()
    df.info()
    df.describe()

    # Creating a text length column
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    # Checking for relations of starts with text length (no relations seen)
    g = sns.FacetGrid(df, col = 'stars')
    g.map(sns.histplot, "text_length")

    # looking at it from an other pov
    sns.boxplot(x='stars', y='text_length',data = df)
    # or via
    sns.countplot(x='stars', data=df)

    ## checking out mean values of numerical colums
    means_by_stars = df.groupby('stars').mean()
    ## looking for correlations:
    corr_means_by_stars = means_by_stars.corr()
    # drawing this
    sns.heatmap(corr_means_by_stars, annot=True)

    ## Grabbing only 1 and 5 start reviews (to make life simpler)
    yelp_class = df[(df['stars'] == 1) | (df['stars'] == 5)]
    # text (X) and stars(y)
    X = yelp_class['text']
    y = yelp_class['stars']

    # Creating a CountVectorizer object
    Cft = CountVectorizer().fit_transform(X) # Convert a collection of text documents to a matrix of token counts and fit and transform it
    stopped here at train test split


