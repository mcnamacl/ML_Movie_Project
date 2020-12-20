import pandas as pd
import numpy as np
import ast

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


def test_file_function(encoded_rows, unique_values, feature_filename):
    feature_file = pd.read_csv(feature_filename, sep="\t", header=None)
    # Gets the array of important features.
    data = np.array(feature_file.iloc[:, 0])
    data = data.reshape(-1, 1)

    indexes = {}
    for i in range(0, len(data)):    
        indexes[data[i][0]] = -1

    index = 0
    for val in unique_values:
        if val in indexes:
            indexes[val] = index
        index = index + 1        

    important_columns = []
    important_column_names = []
    # Transposing rows in order to separate out the values for each new feature.
    encoded_rows = [list(i) for i in zip(*encoded_rows)]
    for val in indexes:
        if indexes[val] != -1:
            important_columns.append(encoded_rows[indexes[val]])
        else:
            important_columns.append(np.zeros(len(encoded_rows[0])))
        important_column_names.append(val)

    # Transposing the data into its original form.
    important_columns = [list(i) for i in zip(*important_columns)]
    return important_columns, important_column_names

# This function hot encodes columns which can contain arrays of values, i.e. multiple values.
def hot_encode_multivals(columns, feature_filename):
    for column in columns:
        unique_values, orig_structure = get_values(column)

        # Number of unique values found for this feature, aka the number of new columns that can be made
        num_new_cols = len(unique_values)

        # Array of one-hot-encoded rows for this feature
        encoded_rows = hot_encode_multival_column(unique_values,
                                                  orig_structure, num_new_cols)

        # If we are limiting the number of new columns to only the most important ones.
        encoded_rows, unique_values = test_file_function(encoded_rows, unique_values, feature_filename)

        add_to_csv(encoded_rows, unique_values)

def hot_encode_single_val_column(unique_values, orig_stucture, len_new_col):
    result = []
    for row in orig_stucture:
        new_row = hot_encode_row(unique_values, row, len_new_col)
        result.append(new_row)
    return result

def hot_encode_single_vals(columns, feature_filename):
    for column in columns:
        unique_values, orig_structure = get_values(column)
        num_new_cols = len(unique_values)
        encoded_rows = hot_encode_single_val_column(unique_values, orig_structure,
                                                num_new_cols)
        encoded_rows, unique_values = test_file_function(encoded_rows, unique_values, feature_filename)
        add_to_csv(encoded_rows, unique_values)

def add_to_csv(data, unique_values):
    data = [list(i) for i in zip(*data)]
    num_columns = len(unique_values)
    for column in range(num_columns):
        column_data = data[column]
        output_data_file[unique_values[column]] = column_data
        output_data_file.to_csv(filename, index=False)

if __name__ == "__main__":
    filename="original_data_2018.csv"
    output_data_file = pd.read_csv(filename, sep="\t", header=None)

    data_columns = [3, 4, 5]
    columns = read_dataset(data_columns)

    data_arrays_important_f = [columns[0]]
    data_arrays = [columns[1]]
    data_single_vals = [columns[2]]

    feature_filenames = ["important_companies.csv", "genres.csv", "langs.csv"]
    hot_encode_multivals(data_arrays_important_f, feature_filenames[0])
    hot_encode_multivals(data_arrays, feature_filenames[1])
    hot_encode_single_vals(data_single_vals, feature_filenames[2])
