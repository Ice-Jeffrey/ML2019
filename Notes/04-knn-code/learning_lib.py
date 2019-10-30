import math


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
        sum += ((u[i] - v[i]) ** 2)
    return math.sqrt(sum)
