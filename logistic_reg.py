import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import eyal_funcs as EB


if __name__ == '__main__':
    udemy_path = EB.basic_path('PC')
    folder = '11-Linear-Regression/'
    file_name = 'none.csv'
    full_path = udemy_path + folder + file_name


