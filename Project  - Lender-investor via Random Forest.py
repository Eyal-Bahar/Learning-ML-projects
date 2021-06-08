import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
import eyal_funcs as EB

## General info :
# Data from LendingClub.com.
# Lending Club connects people who need money (borrowers) with people who have money (investors).
# An investor  would want to invest in a high probability profile for paying you back.
# I will try to create a model that will help predict this.
### Meaning- i will try to predict if a loan has been fully paidback or not ####
#
# Note : Lending club had a very interesting year in 2016, perhaps it would be interesting to keep that context in mind. This data is from before they even went public.
# The csv used in this code has been cleaned of NA values.
#
# Here are what the columns represent:
#
# credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# installment: The monthly installments owed by the borrower if the loan is funded.
# log.annual.inc: The natural log of the self-reported annual income of the borrower.
# dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# fico: The FICO credit score of the borrower.
# days.with.cr.line: The number of days the borrower has had a credit line.
# revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

if __name__ == '__main__':
    df = EB.load_data(folder='15-Decision-Trees-and-Random-Forests/', file_name='loan_data.csv')
    # Explore dataframe
    df.head()
    df.info()
    df.describe()

    ### Visualization of data
    ## histogram for credit policies
    plt.figure()
    df[df['credit.policy'] == 1]['fico'].hist(bins=25)
    df[df['credit.policy'] == 0]['fico'].hist(bins=25, alpha=0.8)
    plt.legend(['Credit policy 1','Credit policy 0'])
    plt.xlabel('fico score')

    ## hist for paid\notpayed loans
    plt.figure()
    df[df['not.fully.paid'] == 1]['fico'].hist(bins=25)
    df[df['not.fully.paid'] == 0]['fico'].hist(bins=25, alpha=0.5)
    plt.legend(['not.fully.paid', 'fully.paid'])
    plt.xlabel('fico score')

    ## counts of loans by purpose, with the color hue defined by not.fully.paid
    sns.countplot(x = df['purpose'], data=df, hue='not.fully.paid')
    plt.title('loan purpose')

    ## The trend between FICO score and interest rate
    plt.figure(4)
    sns.jointplot(x=df['fico'], y=df['int.rate'])

    ## checking if the int.rate vs fico trend differed between no.fully.p[aid and credit.policy
    plt.figure(5)
    sns.lmplot(x='fico', y='int.rate', data=df[df['not.fully.paid']==0], hue='credit.policy')
    plt.title('Fully paid (: ')
    plt.figure(6)
    sns.lmplot(x='fico', y='int.rate', data=df[df['not.fully.paid']==1], hue='credit.policy')
    plt.title('not fully paid = ): ')

    ## Desicion Forest Classification Model

    # transforming purpose to dummy variables
    df_processed = pd.get_dummies(df, columns=['purpose'], drop_first=True)

    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_processed.drop('not.fully.paid', axis=1), df_processed['not.fully.paid'], test_size=0.30, random_state=101 )

    Paid_or_not_tree = DecisionTreeClassifier()
    Paid_or_not_tree.fit(X_train, y_train)

    ## predictions and evaluations
    predictions = Paid_or_not_tree.predict(X_test)

    print('stats of decision tree', '\n')
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print(classification_report(y_test, predictions))

    ## Random Forest Classification Model - Will it perform better?
    # init
    Paid_or_not_RFC = RandomForestClassifier(n_estimators=600)

    # Train
    Paid_or_not_RFC.fit(X_train, y_train) # same train test split as before

    # Evaluate

    predictions_RFC = Paid_or_not_RFC.predict(X_test)

    print('stats of decision tree', '\n')
    print(confusion_matrix(y_test, predictions_RFC))
    print('\n')
    print(classification_report(y_test, predictions_RFC))

    # Notice the recall for each class for the models.
    # Neither did very well, more feature engineering is needed.


