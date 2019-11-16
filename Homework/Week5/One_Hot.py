import numpy as np
import pandas as pd

def onehot(series):
    codes = []
    max = series.max()
    for item in series:
       code = np.zeros(max)
       code[item - 1] = 1
       codes.append(code)
    codes = np.array(codes)
    return codes

def main():
    l = pd.Series([1, 2, 3, 4, 5])
    one_hot_coding = onehot(l)
    print(one_hot_coding)

if __name__ == "__main__":
    main()