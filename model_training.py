import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor 

# Global variables that control what gamma is used in gaussian_kernal.
gamma = [0, 1, 5, 10, 25]
gIndex = 0

# Gaussian kernel to weight neighbours based on their distance from the input feature
def gaussian_kernel(distances):
    g = gamma[gIndex]
    weights = np.exp(-g*(distances**2))
    return weights/np.sum(weights)

# Cross validation for various values of gamma using a kNN model.
def crossValidationkNN(x, y):
    means = []
    devs = []
    gammaVals = [0, 1, 5, 10, 25]
    # Use 5 folds.
    kf = KFold(n_splits=5)
    global gIndex
    gIndex = 0
    for g in gammaVals:
        tmpMeanErrors = []
        for train, test in kf.split(x):
            model = KNeighborsRegressor(n_neighbors=len(x[train]), weights=gaussian_kernel).fit(x[train], y[train])
            ypred = model.predict(x[test])
            # Gets the mean squared error from the predictions vs. the target values.
            predError = mean_squared_error(y[test],ypred)
            # Store the mean squared error.
            tmpMeanErrors.append(predError)
        # Get the mean of the mean squared errors.
        means.append(np.mean(tmpMeanErrors))
        # Get the standard deviation of the mean squared errors.
        devs.append(np.std(tmpMeanErrors))
        gIndex = gIndex + 1
    print("Cross validation gamma: ", means)
    # Plot the mean and standard deviation of various C valies.
    plotMeanAndStdDev(np.array(gammaVals), "Gamma", means, devs, "kNN Cross Validation Gamma")  

# General function for plotting the mean and standard deviation.
def plotMeanAndStdDev(xplot, xlabel, means, devs, title):
    plt.errorbar(xplot,means,yerr=devs)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel('Mean square error')
    plt.show()

def read_dataset():
    df = pd.read_csv("data.csv", header=None)
    X = []
    y = np.array(df.iloc[:,0])
    for column in range(1,len(df.columns)):
        data = np.array(df.iloc[:,column])
        X.append(data)
    return X, y

if __name__ == "__main__":
    X, y = read_dataset()

    X_joined = np.column_stack((X))

    model = KNeighborsRegressor(n_neighbors=len(X), weights=gaussian_kernel).fit(X, y)