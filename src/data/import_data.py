
''' fonctions d’import et d’exploration de données '''
import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_count(df, column, separator):
    """
    Split a column in a DataFrame by a separator and count the number of resulting elements.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to split.
        column (str): The name of the column to split.
        separator (str): The separator to use for splitting.

    Returns:
        pandas.Series: A Series containing the count of elements after splitting.

    """
    return df[column].str.split(separator).str.len()


def split_train_test(data, test_size, train_path="train.csv", test_path="test.csv"):
    """
    Split the data into training and testing sets based on the specified test size.
    Optionally, save the split datasets to CSV files.

    Args:
        data (pandas.DataFrame): The input data to split.
        test_size (float): The proportion of the dataset to include in the test split.
        train_path (str, optional): The file path to save the training dataset.
            Defaults to "train.csv".
        test_path (str, optional): The file path to save the testing dataset.
            Defaults to "test.csv".

    Returns:
        tuple: A tuple containing the training and testing datasets.
    """
    y = data["Survived"]
    X = data.drop("Survived", axis="columns")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    if train_path:
        pd.concat([X_train, y_train]).to_csv(train_path)
    if test_path:
        pd.concat([X_test, y_test]).to_csv(test_path)

    return X_train, X_test, y_train, y_test
