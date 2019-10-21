def createTree(data, features):#label is the name of features!
    classList = [example[-1] for example in data]
    print(classList)
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(data[0]) == 1: #stop splitting when there are no more features in data
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(data)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in data]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(data, bestFeat, value),subLabels)
    return myTree  