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


def partitions(data, feature, thresholds):
    def find_threshold(feature, val):
        # Guaranteed to find a threshold somewhere between min and max
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


def create_thresholds(data, names, nstds=3):
    # Assume the data is normally-distributed
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


def gain(data, H, feature, thresholds):
    ps = partitions(data, feature, thresholds)
    #describe_partitions(ps)
    sum = 0.
    for p in ps.values():
        if feature in p.columns:
            sum += (p.shape[0] / data.shape[0]) * entropy(p)
    return H - sum


def main():
    data = load_data()
    # Split into training and test data sets
    train_data, test_data = train_test_split(data, test_size=0.25)
    # Compute the total entropy for the full data set with respect to the target label
    H = entropy(train_data)
    print(f"Total Entropy: {H}")
    # Generate threshold values for the continuous value descriptive features
    thresholds = create_thresholds(
        train_data, ["age", "chol", "trestbps", "thalach"], nstds=3)
    # Compute the level=0 information gain when partitioned on each descriptive feature
    IG = np.zeros(8)
    for i, feature in enumerate(data.columns[:8]):
        IG[i] = gain(train_data, H, feature, thresholds)
    # Print the best one (at the level=0)
    print(IG)
    print(f"Best IG feature: {data.columns[np.argmax(IG)]}")


if __name__ == "__main__":
    main()
