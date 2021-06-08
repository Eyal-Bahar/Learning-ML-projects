import urllib.request
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import eyal_funcs as EB

def show_iris_types_pictures():
    import urllib
    from PIL import Image

    # The Iris Setosa
    url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
    filename = 'Iris_setosa.jpg'
    urllib.request.urlretrieve(url, filename)
    image = Image.open(filename)
    image.show()
    # The Iris Versicolor
    url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
    filename = 'Iris_versicolor.jpg'
    urllib.request.urlretrieve(url, filename)
    image = Image.open(filename)
    image.show()

    # The Iris Virginica
    url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
    filename = 'Iris_virginica.jpg'
    urllib.request.urlretrieve(url, filename)
    image = Image.open(filename)
    image.show()

if __name__ == '__main__':
    show_iris_types_pictures()

    ## load data
    iris = sns.load_dataset('iris')
    iris.info()
    iris.head()
    ## vizualize
    plt.figure(1)
    sns.pairplot(iris, hue='species') # study connections and find seperation is indeed present straightforward in the data
    plt.figure(2)
    sns.kdeplot(x=iris[iris['species'] == 'setosa']['sepal_width'], y=iris[iris['species'] == 'setosa']['sepal_length'],  cmap="plasma", shade=True)

    ## Train test SPlit
    X_train, X_test, y_train, y_test =  train_test_split(iris.drop('species', axis=1), iris['species'], test_size=0.30, random_state=101 )

    ## init SVC
    model = SVC()
    model.fit(X_train, y_train)

    # predict
    predictions = model.predict(X_test)

    # Evaluation
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions)) # pretty good! But for the sake of practice lets go Gridsearch

    # Creating a grid of params to try possible combinations (Cross-validations)
    param_grid = {'C': [0.1, 1, 1000], 'gamma': [1, 0.01, 0.0001], 'kernel': ['rbf']}
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3) # Creating a grid of params to try possible combinations (Cross-validations)
    grid.fit(X_train, y_train)

    # inspect the best params foun in search
    grid.best_params_
    # inspect the best estimator found in search
    grid.best_estimator_

    # predict with best sestimart
    grid_predictions = grid.predict(X_test)

    # evaluate
    print(confusion_matrix(y_test, grid_predictions))
    print(classification_report(y_test, grid_predictions))

