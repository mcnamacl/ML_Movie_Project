import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor 
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVR
from sklearn.linear_model import Lasso

# Stops dividing by zero kNN error.
np.seterr(divide='ignore', invalid='ignore')

# Global variables that control what gamma is used in gaussian_kernel.
gamma = [0, 1, 5, 10, 25]
gIndex = 0

# Gaussian kernel to weight neighbours based on their distance from the input feature.
def gaussian_kernel(distances):
    g = gamma[gIndex]
    weights = np.exp(-g*(distances**2))
    return weights/np.sum(weights)

# Train a Lasso Regression Model with varying values of C.
def trainWithCombinationsLasso(x, y):
    cValues = [1, 10, 100, 1000, 10000]
    means = []
    print("Lasso Parameters:")

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    for c in cValues:
        model = Lasso(alpha=(1/(2*c))).fit(X_train, y_train)
        print("C=", c)
        # Get the parameter values of the trained model.
        print(model.coef_)
        print(model.intercept_)

        # Use the model to generate predictions.
        newPredictions = model.predict(X_test)

        mean = mean_squared_error(y_test, newPredictions)
        means.append(mean)

        print("Mean Lasso " + str(c))
        print(mean_squared_error(y_test, newPredictions))

    # Plot the mean and standard deviation of various C values.
    plotMeanAndStdDev(np.array(cValues), "C", means, [0,0,0,0,0], "Lasso Cross Validation C")  

# Cross validation for various values of C using a KernelRidge model.
def crossValidationLinearSVR(x, y):
    means = []
    devs = []
    C = [0.01, 0.1, 1, 10, 25]
    # Use 5 folds.
    kf = KFold(n_splits=5)
    for c in C:
        tmpMeanErrors = []
        for train, test in kf.split(x):
            model = LinearSVR(C=c).fit(x[train], y[train])
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
    print("Cross validation c: ", means)
    # Plot the mean and standard deviation of various C valies.
    plotMeanAndStdDev(np.array(C), "C", means, devs, "LinearSVR Cross Validation C")  

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
            model = KNeighborsRegressor(n_neighbors=len(x[train]), weights=gaussian_kernel).fit(
                x[train], y[train])
            
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

# General function for plotting the mean and standard deviation.
def plotMeanAndStdDev(xplot, xlabel, means, devs, title):
    plt.errorbar(xplot,means,yerr=devs)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel('Mean square error')
    plt.show()

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

# Normalising input data, used for arrays that contain very large values.
def normalise_data(input_array):
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
def read_dataset():
    df = pd.read_csv("data.csv")
    X = []
    y = np.array(df.iloc[:,0])
    y = normalise_data(y)
    for column in range(1,len(df.columns)):
        data = np.array(df.iloc[:,column])
        if column == 1:
            data = normalise_data(data)
        X.append(data)
    return X, y

if __name__ == "__main__":
    X, y = read_dataset()

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit_transform(X)

    X_joined = np.column_stack((X))

    testLinearRegression(X_joined, y)

    crossValidationkNN(X_joined, y)

    crossValidationLinearSVR(X_joined, y)

    trainWithCombinationsLasso(X_joined, y)