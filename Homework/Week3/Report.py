import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

def loadData():
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = pd.read_csv('Homework\Week3\pima-indians-diabetes.data.csv', names=names)
    return data

def main():
    # 1. load the data and initial the summary
    data = loadData()    
    
    # 2. for continuous values, compute min, max, mean, standard deviation, first quartile value, second 
    continuous_information = []
    continuous_features = ['plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']
    for feature in continuous_features:
        #initial the subdata
        subdata = data[feature]
        #use the built-in describe method of pandas
        description = subdata.describe()
        #compute missing_values and cardinality of the data
        missing_values = subdata.isnull().sum().sum()
        s = set(subdata)
        cardinality = len(s)
        #store the missing_values and cardinality in a series
        others = [missing_values, cardinality]
        others = pd.Series(others, index = ['missing_values', 'cardinality'])
        #append two series and store in the final result
        description = description.append(others)
        continuous_information.append(description)
    continuous_information = pd.DataFrame(continuous_information, index=continuous_features)
    print('The summary of continuous features: ')
    print(continuous_information, '\n')

    # 3. for categorical values, compute mode, mode frequency, mode percentage, second mode, second mode frequency, second mode percentage
    categorical_information = []
    categorical_features = ['preg', 'class']
    for feature in categorical_features:
        #initial the subdict and subdata
        subdata = data[feature]
        #compute count, missing_values and cardinality
        count = subdata.count()
        missing_values = subdata.isnull().sum().sum()
        s = set(subdata)
        cardinality = len(s)
        #compute the mode
        mode = subdata.mode()[0]
        mode_frequency = 0
        for item in subdata:
            if item == mode:
                mode_frequency += 1
        mode_percentage = mode_frequency / count
        #compute second mode
        subdata_copy = subdata[~subdata.isin([mode])]
        second_mode = subdata_copy.mode()[0]
        second_mode_frequency = 0
        for item in subdata_copy:
            if item == second_mode:
                second_mode_frequency += 1
        second_mode_percentage = second_mode_frequency / count
        information = pd.Series(
            [count, missing_values, cardinality, mode, mode_frequency, mode_percentage, second_mode, second_mode_frequency, second_mode_percentage],
            index=['count', 'missing_values', 'cardinality', 'mode', 'mode_frequency', 'mode_percentage', 'second_mode', 'second_mode_frequency', 'second_mode_percentage']
        )
        categorical_information.append(information)
    categorical_information = pd.DataFrame(categorical_information, index=categorical_features)
    print('The summary of categorical features: ')
    print(categorical_information, '\n')

    # 4. show probability distributions
    data.hist(figsize=(10,10))
    plt.show()    

    # 5. plot the density curves
    data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(10,10))
    plt.show()

    # 6. plot bar plots
    data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10, 10))
    plt.show()

    # 7. show the scatter matrix
    scatter_matrix(data, figsize=(15,15))
    plt.show()

if __name__ == "__main__":
    main()