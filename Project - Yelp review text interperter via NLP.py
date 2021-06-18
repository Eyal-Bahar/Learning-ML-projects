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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline # instead of doing countvactorization,
# transformation of text  to numerics, and then do the TF-IDF  (that i think gives weights
# to words apearing more frequcny - by defintion:
# Term Frequency * log(inverse Document Frequency) = N_term_in_doc / N_tot_in_doc * log(N_docs / N_docs with term_in_them) )
# and then applies the TR-IDF for transform the word numerics to their TF-IDF.
import numpy as np




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

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(Cft, y, test_size = 0.3, random_state = 101)

    # Training and estimating by MultinomialNB
    MND = MultinomialNB()
    MND.fit(X_train, y_train)

    # predictions:
    prediction = MND.predict(X_test)

    # Evaluations:
    print(classification_report(y_test, prediction,))
    print(confusion_matrix(y_test, prediction))

    # Including TF-IDF ( = term frequency–inverse document frequency)
    # a numerical statistic that is intended to reflect how important
    # a word is to a document in a collection or corpus.

    #Create pipeline - doing the following steps:
    Classifiers = [MultinomialNB(), BernoulliNB(), GaussianNB()]
    for classifier in Classifiers:
        pipeline = Pipeline([
        ('bow', CountVectorizer()),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', classifier),  # train on TF-IDF vectors w/ Naive Bayes classifier
            # Classifiers = [MultinomialNB(), BernoulliNB(), GaussianNB]

    ])

        #train:
        X_train, X_test, y_train, y_test = train_test_split(yelp_class['text'], yelp_class['stars'], test_size = 0.3, random_state = 101)
        #fit
        if str(classifier) == 'GaussianNB()':
            npx = np.array(X_train)
            try:
                pipeline.fit(npx, y_train)
            except TypeError:
                print('GaussianNB method is not applicable and raised an error \n : '
                      'A sparse matrix was passed, but dense data is required.')
                continue
        else:
            pipeline.fit(X_train, y_train)
        #predict
        precitions_pipeline = pipeline.predict(X_test)
        # Evaluations:
        print(f'Evaluations for classifier = {classifier}')
        print(classification_report(y_test, precitions_pipeline))
        print(confusion_matrix(y_test, precitions_pipeline))

    print('Conclusion : MultinomialNB() wasnt great, BernoulliNB() was better, gaussian dosnt fit data sparsity '
          'The reason that Bernouli was better is porbabily because its design for binary features and '
          'here the rating has only 2 option - but Bernouli would probably be probalamtic in expanding for a 5 class rating system ')







