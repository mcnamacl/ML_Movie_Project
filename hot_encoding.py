from numpy.core.fromnumeric import ravel
import pandas as pd
import numpy as np
import ast
from sklearn.feature_selection import f_regression
import math, operator

filename = "original_data.csv"
output_data_file = pd.read_csv(filename, sep="\t", header=None)


# Reads in the columns which we want to one-hot-encode.
def read_dataset(columns):
    data = []
    # Gets the array of 'values' for each column.
    for column in columns:
        X = np.array(output_data_file.iloc[:, column])
        X = X.reshape(-1, 1)
        data.append(X)
    # We then drop these columns as we will be replacing them with new ones.
    output_data_file.drop(columns=output_data_file.columns[columns],
                          axis=1,
                          inplace=True)
    output_data_file.to_csv(filename, index=False)
    return data


# Reads in the revenue from the data file, which is our output y.
def read_output():
    y = np.array(output_data_file.iloc[:, 0])
    y = y.reshape(-1, 1)
    return y


# This function gets all of the unique values for a given column,
# as well as the specific values contained in each row of the columns.
def get_values(rows):
    values = set()
    orig_structure = []
    # For each row in the column.
    for row in rows:
        curr_row = []
        inside_row = row[0]
        # If the row contains multiple values.
        if inside_row.startswith("["):
            vals = ast.literal_eval(inside_row)
            row_array = []
            # For each value in this row.
            for value in vals:
                values.add(value)
                row_array.append(value)
            curr_row.append(row_array)
        else:
            values.add(inside_row)
            curr_row.append(inside_row)
        orig_structure.append(curr_row)
    return list(values), orig_structure


def hot_encode_multival_column(unique_values, orig_structure, num_new_cols):
    # Array of one-hot-encoded arrays
    results = []
    # For each array containing an array of multiple values from the original column
    for outer_row in orig_structure:
        inner_row = outer_row[0]
        new_row = hot_encode_row(unique_values, inner_row, num_new_cols)
        results.append(new_row)
    return results


# Returns a list where everything is zero except where value is in the row of values
def hot_encode_row(unique_values, row, len_new_col):
    hot_encode = np.zeros(len_new_col).tolist()
    for i in range(0, len_new_col):
        if unique_values[i] in row:
            hot_encode[i] = 1
    return hot_encode


# This function hot encodes columns which can contain arrays of values, i.e. multiple values.
def hot_encode_multivals(columns, output, check_importance):
    for column in columns:
        unique_values, orig_structure = get_values(column)

        # Number of unique values found for this feature, aka the number of new columns that can be made
        num_new_cols = len(unique_values)

        # Array of one-hot-encoded rows for this feature
        encoded_rows = hot_encode_multival_column(unique_values,
                                                  orig_structure, num_new_cols)

        # If we are limiting the number of new columns to only the most important ones.
        if check_importance:
            encoded_rows, unique_values = get_important_features(
                encoded_rows, output, unique_values)

        add_to_csv(encoded_rows, unique_values)


# Gets the most influential values in a columns, aka the most important features.
def get_important_features(encoded_rows, output, unique_values):
    indexes = getTopFeatures(encoded_rows, output)
    important_columns = []
    important_column_names = []

    # Transposing rows in order to separate out the values for each new feature.
    encoded_rows = [list(i) for i in zip(*encoded_rows)]
    for i in range(0, len(indexes)):
        important_columns.append(encoded_rows[i])
        important_column_names.append(unique_values[i])

    # Transposing the data into its original form.
    important_columns = [list(i) for i in zip(*important_columns)]
    return important_columns, important_column_names


# found here: https://www.programcreek.com/python/example/93975/sklearn.feature_selection.f_regression
def getTopFeatures(train_x, train_y, n_features=10):
    train_x = np.array(train_x)
    train_y = ravel(train_y)
    f_val, p_val = f_regression(train_x, train_y)
    f_val_dict = {}
    p_val_dict = {}
    for i in range(len(f_val)):
        if math.isnan(f_val[i]):
            f_val[i] = 0.0
        f_val_dict[i] = f_val[i]
        if math.isnan(p_val[i]):
            p_val[i] = 0.0
        p_val_dict[i] = p_val[i]

    sorted_f = sorted(f_val_dict.items(),
                      key=operator.itemgetter(1),
                      reverse=True)
    feature_indexs = []
    for i in range(0, n_features):
        feature_indexs.append(sorted_f[i][0])
    return feature_indexs


def hot_encode_single_val_column(unique_values, orig_stucture, len_new_col):
    result = []
    for row in orig_stucture:
        new_row = hot_encode_row(unique_values, row, len_new_col)
        result.append(new_row)
    return result


def hot_encode_single_vals(columns):
    for column in columns:
        unique_values, orig_structure = get_values(column)
        num_new_cols = len(unique_values)
        new_cols = hot_encode_single_val_column(unique_values, orig_structure,
                                                num_new_cols)
        add_to_csv(new_cols, unique_values)


def add_to_csv(data, unique_values):
    data = [list(i) for i in zip(*data)]
    num_columns = len(unique_values)
    for column in range(num_columns):
        column_data = data[column]
        output_data_file[unique_values[column]] = column_data
        output_data_file.to_csv(filename, index=False)


if __name__ == "__main__":
    output = read_output()
    data_columns = [3, 4, 5]
    columns = read_dataset(data_columns)

    data_arrays_important_f = [columns[0]]
    data_arrays = [columns[1]]
    data_single_vals = [columns[2]]

    hot_encode_multivals(data_arrays_important_f, output, True)
    hot_encode_multivals(data_arrays, output, False)
    hot_encode_single_vals(data_single_vals)
