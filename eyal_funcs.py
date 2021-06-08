import pandas as pd

def basic_path(location='PC'):
    if location=='PC':
        return "C:/Users/eyalb/Dropbox/Eyal Bahar/Phd - Tel Aviv/פייתון/udemy/python_for_DS_ML/Py_DS_ML_Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/"
    elif location=='TAU':
        return "D:/Dropbox/Eyal Bahar/Phd - Tel Aviv/פייתון/udemy/python_for_DS_ML/Py_DS_ML_Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/"


def load_data(folder = '13-Logistic-Regression/', file_name = 'advertising.csv', PC_or_TAU = 'PC'):
    ## data path
    data_folder = basic_path(PC_or_TAU)
    full_path = data_folder + folder + file_name
    ## load data
    return pd.read_csv(full_path)  # data
    ## General structure of data_set:
    # 'Daily Time Spent on Site': consumer time on site in minutes
    # 'Age': cutomer age in years
    # 'Area Income': Avg. Income of geographical area of consumer
    # 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
    # 'Ad Topic Line': Headline of the advertisement
    # 'City': City of consumer
    # 'Male': Whether or not consumer was male
    # 'Country': Country of consumer
    # 'Timestamp': Time at which consumer clicked on Ad or closed window
    # 'Clicked on Ad': 0 or 1 indicated clicking on Ad
