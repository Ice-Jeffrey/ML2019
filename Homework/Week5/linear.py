from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib import pyplot as plt
    
def main():
    dataset = load_diabetes()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    #To get the intercept:
    print(f'Intercept: {model.intercept_}')
    #to get the coefficients:
    coeff_df = pd.DataFrame(model.coef_, dataset.feature_names, columns=['Coefficient'])  
    print(coeff_df)

    y_pred = model.predict(X_test)
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

    print(f'Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}')  
    print(f'Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}')  
    print(f'Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')

    subset = df.head(25)
    print(subset)
    
    subset.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

if __name__ == "__main__":
    main()