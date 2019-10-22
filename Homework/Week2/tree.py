import numpy as np
import pandas as pd
import math

def Entropy(data):
    #compute total entropy of the dataset
    counts = data["target"].value_counts()
    """
        Similar to doing the following manually:
            counts = {}
            for val in data["target"]:
                counts[val] = counts.get(val, 0) + 1
    """
    total = data["target"].shape[0]
    sum = 0.
    for count in counts:
        p = count/total
        sum -= p * math.log(p)
    return sum

def create_thresholds(data, names, nstds = 2):
    thresholds = {}
    for feature in names:
        col = data[feature]
        mint, maxt = np.min(col), np.max(col)
        mean, std = np.mean(col), np.std(col)
        ts = [mint]
        for i in range(-nstds, nstds + 1):
            t = round(i * std + mean)
            if t >= mint and t <= maxt:
                ts.append(t)
        ts.append(maxt)
        thresholds[feature] = ts
    return thresholds

def changeData(data, features, thresholds):    
    for feature in features:
        if feature in thresholds:
            for j in range(len(data[feature])):
                for i in range(len(thresholds[feature])-1):
                    val = data[feature][j]
                    if val >= thresholds[feature][i] and val <= thresholds[feature][i+1]:
                        data[feature][j] = i
    return data

def gain(data, feature):
    H = Entropy(data)
    total = data.shape[0]
    vals = {}
    for val in data[feature]:
       vals[val] = vals.get(val, 0) + 1
    entropies = []
    for val in vals:
        temp = Entropy(data[data[feature] == val])
        entropies.append(temp)
    sum = 0.
    for i in range(len(vals)):
        sum += vals[i] * entropies[i] / total
    
    return H - sum


def main():
    data = data = pd.read_csv("Notes\\02-decision-trees-code\heart.csv", usecols=[
                       "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "target"])
    
    thresholds = create_thresholds(
        data, ["age", "chol", "trestbps", "thalach"], nstds=2)
    
    data = changeData(data, features=["age", "chol", "trestbps", "thalach"], thresholds=thresholds)
    print(data)

    print(Entropy(data))
    IG = []
    for feature in data.columns[:-1]:
        temp = gain(data, feature)
        IG.append(temp)
    print(IG)

if __name__ == '__main__':
    main()