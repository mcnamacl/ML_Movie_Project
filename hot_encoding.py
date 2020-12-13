import csv
import pandas as pd
import numpy as np
import ast

filename = "data.csv"
output_data_file = pd.read_csv(filename, sep="\t")


def read_dataset(columns):
    data = []
    for column in columns:
        X = np.array(output_data_file.iloc[:, column])
        X = X.reshape(-1, 1)
        data.append(X)
    output_data_file.drop(columns=output_data_file.columns[columns], axis=1, inplace=True)
    output_data_file.to_csv(filename, index=False)
    return data


def get_values(rows):
    values = set()
    orig_structure = []
    for row in rows:
        tmp_row = []
        for inner in row:
            if inner.startswith("["):
                vals = ast.literal_eval(inner)
                tmp_tmp_row = []
                for value in vals:
                    values.add(value)
                    tmp_tmp_row.append(value)
                tmp_row.append(tmp_tmp_row)
            else:
                values.add(inner)
                tmp_row.append(inner)
        orig_structure.append(tmp_row)
    return list(values), orig_structure


def gen_array_hot_encode_rows(unique_values, orig_stucture, len_new_col):
    result = []
    for row_deep in orig_stucture:
        for row in row_deep:
            new_row = gen_hot_encode_row(unique_values, row, len_new_col)
            result.append(new_row)
    return result


def gen_hot_encode_row(unique_values, row, len_new_col):
    hot_encode = np.zeros(len_new_col).tolist()
    for i in range(0, len_new_col):
        if unique_values[i] in row:
            hot_encode[i] = 1
    return hot_encode


def hot_encode_arrays(data):
    for rows in data:
        unique_values, orig_structure = get_values(rows)

        len_new_col = len(unique_values)

        new_cols = gen_array_hot_encode_rows(unique_values, orig_structure, len_new_col)

        add_to_csv(new_cols, unique_values)


def gen_single_vals_hot_encode_rows(unique_values, orig_stucture, len_new_col):
    result = []
    for row in orig_stucture:
        new_row = gen_hot_encode_row(unique_values, row, len_new_col)
        result.append(new_row)
    return result


def hot_encode_single_vals(data):
    for rows in data:
        unique_values, orig_structure = get_values(rows)

        len_new_col = len(unique_values)

        new_cols = gen_single_vals_hot_encode_rows(
            unique_values, orig_structure, len_new_col
        )

        add_to_csv(new_cols, unique_values)


def add_to_csv(data, unique_values):
    data = [list(i) for i in zip(*data)]

    num_columns = len(unique_values)
    for column in range(num_columns):
        column_data = data[column]
        output_data_file[unique_values[column]] = column_data
        output_data_file.to_csv(filename, index=False)


if __name__ == "__main__":
    array_columns = [3, 4]
    single_val_columns = [5]

    data_columns = [3, 4, 5]

    data = read_dataset(data_columns)

    data_arrays = []
    data_arrays.append(data[0])
    data_arrays.append(data[1])

    data_single_vals = []
    data_single_vals.append(data[2])

    hot_encode_arrays(data_arrays)
    hot_encode_single_vals(data_single_vals)
