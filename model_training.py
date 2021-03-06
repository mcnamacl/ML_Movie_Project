import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor 
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso

# Stops dividing by zero kNN error.
np.seterr(divide='ignore', invalid='ignore')

# Global variables that control what gamma is used in gaussian_kernel.
gamma = [0, 1, 2, 5, 10, 15]
gIndex = 0

# Gaussian kernel to weight neighbours based on their distance from the input feature.
def gaussian_kernel(distances):
    g = gamma[gIndex]
    weights = np.exp(-g*(distances**2))
    return weights/np.sum(weights)

# General function for plotting the mean and standard deviation.
def plotMeanAndStdDev(xplot, xlabel, means, devs, title):
    plt.errorbar(xplot,means,yerr=devs)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel('Mean square error')
    plt.show()

# Train a Lasso Regression Model with varying values of C.
def trainWithCombinationsLasso(x, y):
    cValues = [1, 10, 100, 1000, 10000]
    means = []
    devs = []
    kf = KFold(n_splits=5)

    for c in cValues:
        tmpMeanErrors = []
        for train, test in kf.split(x):
            model = Lasso(alpha=(1/(2*c))).fit(x[train], y[train])

            # Use the model to generate predictions.
            newPredictions = model.predict(x[test])

            mean = mean_squared_error(y[test], newPredictions)
            tmpMeanErrors.append(mean)

        print("Mean Lasso " + str(c))
        print(np.mean(tmpMeanErrors))
        # Get the mean of the mean squared errors.
        means.append(np.mean(tmpMeanErrors))
        # Get the standard deviation of the mean squared errors.
        devs.append(np.std(tmpMeanErrors))

    # Plot the mean and standard deviation of various C values.
    plotMeanAndStdDev(np.array(cValues), "C", means, devs, "Lasso Cross Validation C")  

# Cross validation for various values of gamma using a kNN model.
def crossValidationkNN(x, y):
    means = []
    devs = []
    gammaVals = [0, 1, 2, 5, 10, 15]
    # Use 5 folds.
    kf = KFold(n_splits=5)
    global gIndex
    gIndex = 0
    for g in gammaVals:
        tmpMeanErrors = []
        for train, test in kf.split(x):
            model = KNeighborsRegressor(n_neighbors=int(math.sqrt(len(x[train]))), weights=gaussian_kernel).fit(x[train], y[train])
            ypred = model.predict(x[test])

            mask = np.isnan(ypred)
            idx = np.where(~mask,np.arange(mask.size),0)
            np.maximum.accumulate(idx, out=idx)
            ypred = ypred[idx]
            
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
    # Plot the mean and standard deviation of various C values.
    plotMeanAndStdDev(np.array(gammaVals), "Gamma", means, devs, "kNN Cross Validation Gamma")  

# Testing linear regression mean squared errors.
def testLinearRegression(x, y):
    # Mean for Linear Regression model using all of the data for training.
    model = LinearRegression().fit(x, y)
    ypred = model.predict(x)  
    lgError = mean_squared_error(y,ypred)

    # Average mean when using k-folds for training and testing.
    kFoldError = kFoldLinearRegression(x,y)

    # Mean when using dummy model with all of the data used for training.
    dummy = DummyRegressor(strategy="mean").fit(X=x, y=y)
    dummy_preds = dummy.predict(x)
    dummyError = mean_squared_error(y,dummy_preds)

    # Mean when using an 80:20 train:test split for Linear Regression.
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    splitError = mean_squared_error(y_test,preds)

    print("LG k-fold mean squared error: %f, baseline square error: %f, LG not-folded error: %f, LG 80:20 error: %f"
    %(kFoldError, dummyError, lgError, splitError))

# Cross validation for various values of gamma using a Linear Regression model.
def kFoldLinearRegression(x, y):
    kf = KFold(n_splits=5)
    meanErrors = []
    for train, test in kf.split(x):
        model = LinearRegression().fit(x[train], y[train])
        ypred = model.predict(x[test])       

        # Gets the mean squared error from the predictions vs. the target values.
        predError = mean_squared_error(y[test],ypred)
        # Store the mean squared error.
        meanErrors.append(predError)

    return np.mean(meanErrors)

def testEachModel(X, y, X_test, y_test):
    lasso_model = Lasso(alpha=(1/(2*1000))).fit(X, y)
    lasso_preds = lasso_model.predict(X_test)
    lasso_predError = mean_squared_error(lasso_preds, y_test)
    print(lasso_model.coef_)
    print("Lasso mean squared error: ", lasso_predError)

    global gIndex
    global gamma
    gamma = [1]
    gIndex = 0
    kNN_model = KNeighborsRegressor(n_neighbors=len(X), weights=gaussian_kernel).fit(X, y)
    kNN_preds = kNN_model.predict(X_test)
    kNN_predError = mean_squared_error(kNN_preds, y_test)
    print("kNN mean squared error: ", kNN_predError)

    lg_model = LinearRegression().fit(X, y)
    lg_preds = lg_model.predict(X_test)
    lg_predError = mean_squared_error(lg_preds, y_test)
    print(lg_model.coef_)
    print("Linear Regression mean squared error: ", lg_predError)

    # Mean when using dummy model with all of the data used for training.
    dummy = DummyRegressor(strategy="mean").fit(X=X, y=y)
    dummy_preds = dummy.predict(X_test)
    dummyError = mean_squared_error(y_test,dummy_preds)
    print("Dummy Regressor mean squared error: ", dummyError)

# Normalising input data, used for arrays that contain very large values.
def normaliseData(input_array):
    # This function normalises an array's values using the formula: 
    # new_value = (curr_value - min_value) / (max_value - min_value)
    min_value = min(input_array)
    max_value = max(input_array)
    output_array = []
    for value in input_array:
        output_array.append(float((value - min_value)) / (max_value - min_value))
    output_array = np.array(output_array)   
    return output_array

# Reads in dataset and creates y, the output, and X, an array of arrays
# where each array is a feature column. 
def readDataset(filename):
    df = pd.read_csv(filename)
    X = []
    y = np.array(df.iloc[:,0])
    y = normaliseData(y)
    for column in range(1,len(df.columns)):
        data = np.array(df.iloc[:,column])
        if column == 1 or column == 3:
            data = normaliseData(data)
        X.append(data)
    return X, y

if __name__ == "__main__":
    X, y = readDataset("CSV_files/final_training.csv")

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit_transform(X)

    X_joined = np.column_stack((X))

    testLinearRegression(X_joined, y)

    crossValidationkNN(X_joined, y)

    trainWithCombinationsLasso(X_joined, y)

    X_test, y_test = readDataset("CSV_files/final_testing.csv")

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit_transform(X_test)

    X_test_joined = np.column_stack((X_test))

    testEachModel(X_joined, y, X_test_joined, y_test)
