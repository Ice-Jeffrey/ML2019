import numpy as np
import pandas as pd
import math
import operator
from sklearn.metrics import accuracy_score
from learning_lib import train_test_split

def Gini(data):
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
        p = count / total
        sum -= p ** 2
    return 1 - sum

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
    H = Gini(data)
    total = data.shape[0]
    vals = {}
    for val in data[feature]:
       vals[val] = vals.get(val, 0) + 1
    #print(data)
    #print(vals)
    
    sum = 0.
    for val in vals:
        p = vals[val] / total
        temp = Gini(data[data[feature] == val])
        sum += p * temp
    #print(H, sum)
    #print(H - sum)
    return H - sum

def findBestFeature(features, IG):
    bestLoc = 0
    bestFeature = features[0]
    gain = IG[0]
    for i in range(len(features)):
        if IG[i] > gain:
            bestLoc = i
            bestFeature = features[i]
            gain = IG[i]
    return bestLoc, bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    #print(classCount.items)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDecisionTree(data, features):
    #the ending conditions of the recursion    
    classList = list(data.iloc[:,-1])
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if data.shape[1] == 1:
        return majorityCnt(classList)

    #compute the information gain
    IG = []
    for feature in features:
        temp = gain(data, feature)
        IG.append(temp)
    #print(IG)

    #find the feature with the highest entropy    
    bestloc, bestfeat = findBestFeature(features, IG)
    #print(bestfeat)
    #initial the decision tree
    myTree = {bestfeat: {}}
    #delete the current best feature
    features = features.drop(bestfeat)
    featValues = data[bestfeat]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subFeatures = features
        subData = data[data[bestfeat] == value].drop(bestfeat, axis=1)
        myTree[bestfeat][value] = createDecisionTree(subData, subFeatures)
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = 0
    for i in range(featLabels.shape[0]):
        if featLabels[i] == firstStr:
            featIndex = i
    
    key = testVec[featIndex]
    valueOfFeat = secondDict.get(key, 0)
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


def main():
    # 1. load data
    data = pd.read_csv("Notes\\02-decision-trees-code\\heart.csv")
    for i in range(data.shape[0]):
        data['oldpeak'].iloc[i] = int(math.floor(data['oldpeak'].iloc[i]))

    # 2. create thresholds
    thresholds = create_thresholds(
        data, ["age", "chol", "trestbps", "thalach"], nstds=2)
    # 3. modify the data and transform it into nominal values
    data = changeData(data, features=["age", "chol", "trestbps", "thalach"], thresholds=thresholds)

    # 4. split the dataset to be training data and testing data
    train_data, test_data = train_test_split(data, test_size=0.25)
    
    # 5. create the decision tree
    myTree = createDecisionTree(train_data.drop('index', axis = 1), data.columns[:-1])
    
    # 6. make the prediction based on the decision tree we have built
    predictions = []
    for i in range(test_data.shape[0]):
        row = list(test_data.iloc[i, 1:])
        result = classify(myTree, data.columns[:-1], row)
        predictions.append(result)

    print(accuracy_score(test_data.iloc[:, -1], predictions))
    

if __name__ == '__main__':
    main()