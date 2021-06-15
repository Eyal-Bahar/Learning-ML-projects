import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import eyal_funcs as EB
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



def remove_punc_and_stopwords(msg):
    """
    :param msg: recieved 'msg' : a string we would like to process
    :return: a list of words cleaned from puncutation and stopwords

    1: Removes punctuation marks ( # , ! etc..)

    2: Remove stopwords, common english words we believe irrelevant for our analysis

    """
    msg_list = [letter for letter in msg if letter not in string.punctuation]
    msg_no_punc = ''.join(msg_list)
    return [word for word in msg_no_punc.split() if word.lower() not in stopwords.words('english')] # stopwords.words('english')  # common words the are probably not helpfull for spam or ham


if __name__ == '__main__':
    ## Spam vs Ham via NLP

    # nltk.download_shell() #download 'stopwords' only if needed
    # stop words are common words such as the a etc..

    ## loading SMS span collection see read me in loadidng folder
    # The SMS Spam Collection v.1 (hereafter the corpus) is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.
    load_path = EB.basic_path(location='PC') + '20-Natural-Language-Processing/' + 'smsspamcollection/SMSSpamCollection'
    messages = [line.rstrip() for line in open(load_path)]
    print(len(messages))

    # check out first 10 msg's
    for msg_num, msg in enumerate(messages[:10]):
        print(msg_num, msg)
        print('\n')

    # read data with pandas and recognize the tab separator indicating type before the actual msg :
    messages = pd.read_csv(load_path, sep = '\t', header=None)
    messages.columns = ['label', 'message']
    # inspect datafrane
    messages.head()

    # exploring the data and vizualization
    messages.describe()
    messages.groupby('label').describe()
    messages['length'] = messages['message'].apply(len)

    ##
    plt.figure(1)
    messages['length'].hist(bins=50)

    ## applying the "Bag of words" method to convered words to numerics(vecotrs) in order to used ML tools
    # cleaning messeges for punctuation and common words and then
    # create a table of word \ appearence in msg
    # might wanna be using a spars matrix to save memory
    bow_transformer = CountVectorizer(analyzer=remove_punc_and_stopwords).fit(messages['message']) # might take some time
    print(len(bow_transformer.vocabulary_)) # 11425

    ## exmaple of transformer :
    example4 = bow_transformer.transform([messages['message'][3]])
    print(example4) # showing the encoded words id and its reaacurence
    # you can use get_feature_names to desipher the "code"
    bow_transformer.get_feature_names()[9554]

    ## Transform all msg's
    messages_bow = bow_transformer.transform(messages['message'])
    print('Shape of Sparse Matrix: ', messages_bow.shape)
    print('Amount of Non-Zero occurrences: ', messages_bow.nnz)
    ##
    sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
    print('sparsity: {}'.format((sparsity)))

    ## wegiht and normilization
    #example
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    tfidf_example4 = tfidf_transformer.transform(example4)
    print(tfidf_example4)

    # all of it
    messages_tfidf = tfidf_transformer.transform(messages_bow)






