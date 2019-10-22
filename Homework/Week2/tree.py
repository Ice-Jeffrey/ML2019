import pandas as pd
import numpy as np
import math
import pprint
from learning_lib import train_test_split

pp = pprint.PrettyPrinter(indent=4)

def load_data():
    # Only include the first 8 descriptive features and the target label
    data = pd.read_csv("Notes\\02-decision-trees-code\heart.csv", usecols=[
                       "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "target"])
    return data

def describe_partitions(ps):
    for target, p in sorted(ps.items(), key=lambda k: k[0]):
        print(f"{target}\t{p.shape[0]}")
    print("")

def entropy(data):
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
        sum += p * math.log(p)
    return - sum

def create_thresholds(data, names, nstds=3):
    # Assume the data is normally-distributed, split values of features into different intervals
    thresholds = {}
    for feature in names:
        col = data[feature]
        mint, maxt = np.min(col), np.max(col)
        mean, stddev = np.mean(col), np.std(col)
        ts = [mint]
        for n in range(-nstds - 1, nstds):
            t = round(n * stddev + mean)
            if t >= mint and t <= maxt:
                ts.append(t)
        thresholds[feature] = ts
    return thresholds

def partitions(data, feature, thresholds):
    def find_threshold(feature, val):
        # Guaranteed to find a threshold somewhere between min and max and locate the exact interval for current data
        for t in reversed(thresholds[feature]):
            if val >= t:
                return t
        raise Exception("Unexpected return without threshold")

    features = data.columns
    ps = {}
    for j, val in enumerate(data[feature]):
        # Treat categorical and continuous feature values differently
        if feature in thresholds:
            val = find_threshold(feature, val)
        p = ps.get(val, pd.DataFrame(columns=features))
        ps[val] = p.append(data.loc[j, features])
    return ps

def gain(data, H, feature, thresholds):
    ps = partitions(data, feature, thresholds)
    #describe_partitions(ps)
    sum = 0.
    for p in ps.values():
        if feature in p.columns:
            sum += (p.shape[0] / data.shape[0]) * entropy(p)
    return H - sum

def majorityCnt(classList):
    """
    majorityCnt(筛选出现次数最多的分类标签名称)

    Args:
        classList 类别标签的列表
    Returns:
        sortedClassCount[0][0] 出现次数最多的分类标签名称
        
    假设classList=['yes', 'yes', 'no', 'no', 'no']    
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]= 0
        classCount[vote] += 1
        """
        print(classCount[vote])的结果为:
        {'yes': 1}
        {'yes': 2}
        {'yes': 2, 'no': 1}
        {'yes': 2, 'no': 2}
        {'yes': 2, 'no': 3}
        """
    sortedClassCount =sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    """
    print(sortedClassCount)的结果为:
    [('no', 3), ('yes', 2)]
    """
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    createTree(创建树)

    Args:
        dataSet 数据集
        labels  标签列表:标签列表包含了数据集中所有特征的标签。最后代码遍历当前选择
    Returns:
        myTree 标签树:特征包含的所有属性值，在每个数据集划分上递归待用函数createTree()，
        得到的返回值将被插入到字典变量myTree中，因此函数终止执行时，字典中将会嵌套很多代
        表叶子节点信息的字典数据。
    """
    labels = dataSet.columns
    #取得dataSet的最后一列数据保存在列表classList中
    classList = dataSet.iloc[:, -1]
    #如果classList中的第一个值在classList中的总数等于长度,也就是说classList中所有的值都一样
    #也就等价于当所有的类别只有一个时停止
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #当数据集中没有特征可分时也停止
    if len(dataSet[0])==1:
        #通过majorityCnt()函数返回列表中最多的分类
        return majorityCnt(classList)
    #通过chooseBestFeatTopSplit()函数选出划分数据集最佳的特症
    bestFeat = chooseBestFeatTopSplit(dataSet) 
    #最佳特征名 = 特征名列表中下标为bestFeat的元素
    bestFeatLabel=labels[bestFeat]
    # 构造树的根节点，多级字典的形式展现树，类似多层json结构
    myTree={bestFeatLabel:{}}
    # 删除del列表labels中的最佳特征(就在labels变量上操作)
    del(labels[bestFeat])
    #取出所有训练样本最佳特征的值形成一个list
    featValues = [example[bestFeat] for example in dataSet]
    # 通过set函数将featValues列表变成集合,去掉重复的值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        #复制类标签并将其存储在新列表subLabels中
        subLabels = labels[:] 
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

def main():
    data = load_data()
    # Split into training and test data sets
    train_data, test_data = train_test_split(data, test_size=0.25)
    # Compute the total entropy for the full data set with respect to the target label
    H = entropy(train_data)
    print(f"Total Entropy: {H}")
    # Generate threshold values for the continuous value descriptive features
    thresholds = create_thresholds(train_data, ["age", "chol", "trestbps", "thalach"], nstds=3)
    # Compute the level=0 information gain when partitioned on each descriptive feature
    IG = np.zeros(8)
    for i, feature in enumerate(data.columns[:8]):
        IG[i] = gain(train_data, H, feature, thresholds)
    # Print the best one (at the level=0)
    print(IG)
    print(f"Best IG feature: {data.columns[np.argmax(IG)]}")


if __name__ == "__main__":
    main()