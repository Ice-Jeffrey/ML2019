import numpy as np
import pandas as pd
import math
import operator
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from learning_lib import train_test_split

def loadData():
    names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age', 'Class']
    data = pd.read_csv('Homework\Week3\pima-indians-diabetes.data.csv', names=names)
    return data

def entropy(data):
    #compute total entropy of the dataset
    counts = data["Class"].value_counts()
    """
        Similar to doing the following manually:
            counts = {}
            for val in data["Class"]:
                counts[val] = counts.get(val, 0) + 1
    """
    total = data["Class"].shape[0]
    sum = 0.
    for count in counts:
        p = count / total
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
    H = entropy(data)
    total = data.shape[0]
    vals = {}
    for val in data[feature]:
       vals[val] = vals.get(val, 0) + 1
    #print(data)
    #print(vals)
    
    sum = 0.
    for val in vals:
        p = vals[val] / total
        temp = entropy(data[data[feature] == val])
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
    H = entropy(data)
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
    valueOfFeat = secondDict.get(key, None)
    if valueOfFeat != None and isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    elif valueOfFeat == None:   classLabel = 0
    else: classLabel = valueOfFeat
    return classLabel

def CreateRandomForest(data, features, classifiers = 10):
    trees = []
    for i in range(classifiers):
        subdata = data.iloc[:,:]
        #bootstraping
        cols = np.random.randint(0, 2, size = (len(features)))
        subfeatures = []
        for j in range(len(cols)):
            if cols[j] == 0:
                subdata = subdata.drop(features[j], axis = 1)
            if cols[j] == 1:
                subfeatures.append(features[j])
        
        #subspace sampling
        subdata = subdata.sample(frac = 0.7)
        myTree = createDecisionTree(subdata, features=subdata.columns[:-1])
        trees.append(myTree)
    return trees

def main():
    # 1. load the data and initial the summary
    data = loadData()    
    
    # 2. data cleaning
    for i in range(data.shape[0]):
        subdata = data.iloc[i]
        if(subdata['Glucose'] == 0):
            subdata['Glucose'] = data[data['Glucose'] != 0]['Glucose'].mean()
            data.iloc[i] = subdata
        if(subdata['BloodPressure'] == 0):
            subdata['BloodPressure'] = data[data['BloodPressure'] != 0]['BloodPressure'].mean()
            data.iloc[i] = subdata
        if(subdata['SkinThickness'] == 0):
            subdata['SkinThickness'] = data[data['SkinThickness'] != 0]['SkinThickness'].mean()
            data.iloc[i] = subdata
        if(subdata['BMI'] == 0):
            subdata['BMI'] = data[data['BMI'] != 0]['BMI'].mean()
            data.iloc[i] = subdata

    # 3. show the correlation and drop the correlated features
    print(data.drop('Class', axis=1).corr())
    data.drop('SkinThickness', axis=1)
    data.drop('Pregnancies', axis=1)

    # 4. create thresholds
    thresholds = create_thresholds(
        data, ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DPF', 'Age'], nstds=2)

    # 5. modify the data and transform it into nominal values
    data = changeData(data, features=['Age', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DPF'], thresholds=thresholds)

    # 6. split the dataset to be training data and testing data
    train_data, test_data = train_test_split(data, test_size=0.25)
    
    # 7. create the decision tree
    myTree = createDecisionTree(train_data.drop('index', axis = 1), data.columns[:-1])
    #print(myTree)

    # 8. make the prediction based on the decision tree we have built
    decision_tree_predictions = []
    for i in range(test_data.shape[0]):
        row = list(test_data.iloc[i, 1:])
        result = classify(myTree, data.columns[:-1], row)
        decision_tree_predictions.append(result)
    # print the accuracy
    print('The accuracy using decision tree is: ', accuracy_score(test_data.iloc[:, -1], decision_tree_predictions))

    # 9. make the prediction based on the random forest we have built
    myForest = CreateRandomForest(train_data.drop('index', axis = 1), data.columns[:-1])
    random_forest_predictions = []
    for i in range(test_data.shape[0]):
        results = {}
        for myTree in myForest:
            row = list(test_data.iloc[i, 1:])
            result = classify(myTree, data.columns[:-1], row)
            results[result] = results.get(result, 0) + 1
        max_key = None
        for key, value in results.items():
            if value == max(results.values()):
                max_key = key
        random_forest_predictions.append(max_key)
    # compute the accuracy_score of the decision tree
    print('The accuracy using random forest is: ', accuracy_score(test_data.iloc[:, -1], random_forest_predictions))

if __name__ == "__main__":
    main()