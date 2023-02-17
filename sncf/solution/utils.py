import pandas as pd
import numpy as np
import csv
import os
from sklearn.model_selection import train_test_split

__author__ = "Pierre Jaumier"
__version__ = "Data IA, LSdO, février 2023"


def get_data(relative_path='../data/'):
    x_train = pd.read_csv(os.path.join(relative_path, 'x_train.csv'))
    y_train = pd.read_csv(os.path.join(relative_path, 'y_train.csv'))
    x_test = pd.read_csv(os.path.join(relative_path, 'x_test.csv'))

    # Dataset avec cible incluse
    df = x_train.copy()
    df['p0q0'] = y_train.p0q0
    return x_train, y_train, df, x_test


def split_data(df, chosen_columns):
    x_train, x_test, y_train, y_test = train_test_split(df[chosen_columns],
                                                        df[['p0q0']], test_size=0.2, random_state=42)

    df_train = x_train.copy()
    df_train['p0q0'] = y_train.p0q0

    df_test = x_test.copy()
    df_test['p0q0'] = y_test.p0q0

    return x_train, y_train, df_train, x_test, y_test, df_test


def create_submission_csv(y_test, filename):
    """Création du fichier csv pour soumission au challenge

    Args:
        y_test (list): liste des solutions de dimension len(x_test)
        filename (str): nom du fichier sans extension qui sera sauvegardé
                        dans ../data/prediction/filename.csv
    """
    df = pd.DataFrame(y_test, columns=['p0q0'])
    df.index += 1
    df.index = df.index.map(str)
    filename = filename + '.csv'
    df.to_csv(os.path.join('../data/prediction/', filename), quoting=csv.QUOTE_NONNUMERIC)
