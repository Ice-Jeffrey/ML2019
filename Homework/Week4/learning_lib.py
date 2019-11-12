import math
import numpy as np
import pandas as pd

class Node():
    def __init__(self, data = None, label = None, left = None, right = None, feature = None):
        self.data = data
        self.label = label
        self.left = left
        self.right = right
        self.feature = feature

def distance_metric(u, v):
    """
    Compute a distance metric between two feature vectors u and v
    using n-dimensional Euclidean distance
    """
    if len(u) != len(v):
        raise Exception(
            "Distance metric not valid for differently sized vectors")
    sum = 0.
    for i in range(len(u)):
        sum += (u[i] - v[i]) ** 2
    return math.sqrt(sum)

def createKDTree(X_train, y_train):
    # 1. judge if the recursion ends
    if(X_train.shape[0] == 0):
        return None
    if(X_train.shape[0] == 1):
        return Node(X_train.iloc[0,:], y_train.iloc[0])

    # 2. find the feature with the largest variance
    vars = []
    for i in range(X_train.shape[1]):
        vars.append(X_train.iloc[:, i].var())
    max_var = X_train.columns[np.argmax(np.array(vars))]
    
    # 3. find the median between the max feature
    data = sorted(X_train[max_var])
    median = data[round(len(data) / 2)]
    med_index = -1    
    
    # 4. build leftdata and rightdata
    for i in range(len(X_train)):
        if X_train.iloc[i, : ][max_var] == median:
            med_index = i
            break

    leftdata = X_train.iloc[0:med_index, :]
    leftlabel = y_train.iloc[0:med_index]
    rightdata = X_train.iloc[med_index + 1 : , ]
    rightlabel = y_train.iloc[med_index + 1 : ]

    # 5. build the kd-tree
    node = Node(X_train.iloc[med_index, :], y_train.iloc[med_index], createKDTree(leftdata, leftlabel), createKDTree(rightdata, rightlabel), max_var)
    return node

def Depth(tree):
    if tree == None:
        return 0
    return 1 + max(Depth(tree.left), Depth(tree.right))

def Traverse(tree):
    if tree == None:
        return None
    print(tree.label)
    Traverse(tree.left)
    Traverse(tree.right)

def findLeaf(tree, data):
    if tree == None:
        return None
       
    # get the path for the whole routing
    path = []
    root = tree
    count = 0
    while root != None:
        path.append(root)
        feature = root.feature
        if feature != None and root.data[feature] < data[feature]:
            root = root.right
        elif feature != None and root.data[feature] > data[feature]:
            root = root.left
        else:
            break

    # assume that the leaf node is the nearest node
    nearest = path[-1]
    a = np.array(nearest.data).tolist()
    if len(a) == 1:
        a = a[0]
    b = np.array(data).tolist()
    if len(b) == 1:
        b = b[0]
    min_distance = distance_metric(a, b)

    # trace_back
    while path != []:
        backpoint = path.pop()
        if backpoint != None:
            feature = backpoint.feature
            kd_point = None
            a = np.array(backpoint.data).tolist()
            if len(a) == 1:
                a = a[0]
            b = np.array(data).tolist()
            if len(b) == 1:
                b = b[0]
            if distance_metric(a, b) < min_distance:
                if data[feature] < backpoint.data[feature]:
                    kd_point = backpoint.right
                else:
                    kd_point = backpoint.left
                path.append(kd_point)
            if kd_point != None:
                c = np.array(kd_point.data).tolist()
                if len(c) == 1:
                    c = c[0]
                if min_distance > distance_metric(c, b):
                    nearest = kd_point
                    min_distance = distance_metric(c, b)
    
    return [nearest, min_distance]