import pandas as pd
import numpy as np
import csv
import os

__author__ = "Pierre Jaumier"
__version__ = "Data IA, LSdO, février 2023"


def define_column_data_type(prefix, data_type, column_count, column_type):
    """Fonction d'aide pour la création des associations des colonnes à leurs types
    """
    for i in range(1, column_count+1):
        column_name = prefix + str(i)
        column_type[column_name] = data_type


def get_data(relative_path='../data/'):

    column_type = {}  # Dictionnaire qui à un nom de colonne associe son type
    column_type['ID'] = 'int'
    define_column_data_type('item', 'string', 24, column_type)
    define_column_data_type('cash_price', 'float', 24, column_type)
    define_column_data_type('make', 'string', 24, column_type)
    define_column_data_type('model', 'string', 24, column_type)
    define_column_data_type('goods_code', 'string', 24, column_type)
    define_column_data_type('Nbr_of_prod_purchas', 'float', 24, column_type)
    # pas possible d'utiliser des int car pandas n'accepte pas les nan avec des int
    column_type['Nb_of_items'] = 'float'

    x_train = pd.read_csv(os.path.join(relative_path, 'x_train.csv'), dtype=column_type)
    x_test = pd.read_csv(os.path.join(relative_path, 'x_test.csv'), dtype=column_type)
    y_train = pd.read_csv(os.path.join(relative_path, 'y_train.csv'), index_col=False, usecols=['fraud_flag'])

    # Renommage des colonnes
    # id : identifiant du panier
    mapper = {}
    mapper['ID'] = 'id'
    # category_i : une catégorie de produits dans le panier (1 <= i <= 24)
    mapper.update({'item' + str(i): 'category_' + str(i) for i in range(1, 25)})
    # price_i : le prix d'un produit de type i
    mapper.update({'cash_price' + str(i): 'price_' + str(i) for i in range(1, 25)})
    # manufacturer_i : fabriquant
    mapper.update({'make' + str(i): 'manufacturer_' + str(i) for i in range(1, 25)})
    # product_i : libellé du produit de type i
    mapper.update({'model' + str(i): 'product_' + str(i) for i in range(1, 25)})
    # product_code_i : code  du produit de type i
    mapper.update({'goods_code' + str(i): 'product_code_' + str(i) for i in range(1, 25)})
    # product_count_i : nombre de produits d'une certain type dans le panier
    mapper.update({'Nbr_of_prod_purchas' + str(i): 'product_count_' + str(i) for i in range(1, 25)})
    # product_type_count : nombre de types de produits dans le panier, valeur comprise entre 1 et 24
    # et qui renseigne le nombre de colonnes significatives
    mapper['Nb_of_items'] = 'product_type_count'
    # fraud : booléen inidiquant s'il y a eu fraude ou non
    mapper['fraud_flag'] = 'fraud'

    x_train = x_train.rename(columns=mapper)
    x_test = x_test.rename(columns=mapper)
    y_train = y_train.rename(columns=mapper)

    # Dataset avec cible incluse
    df = x_train.copy()
    df['fraud'] = y_train.fraud
    return x_train, y_train, df, x_test


def distinct_values_for(column_name_prefix, df):
    """Renvoie la liste des valeurs uniques apparaissant dans un ensemble de colonnes ayant le même préfixe
    """
    s = set()
    for i in range(1, 25):
        column_name = column_name_prefix + str(i)
        s = s.union(set(list(df[column_name].unique())))
    return list(s)


def list_unique_values_from_text_columns(df):
    """Listes des occurences uniques de valeurs textuelles pour les différents types de colonnes textuelles
    category_
    manufacturer_
    product_
    product_code_
    """
    categories = distinct_values_for('category_', df)
    manufacturers = distinct_values_for('manufacturer_', df)
    products = distinct_values_for('product_', df)
    product_codes = distinct_values_for('product_code_', df)

    return categories, manufacturers, products, product_codes


def transform_prefixed_columns_in_series(prefix, df):
    """Tranformation du dataframe pour un type de préfixe en une seule série de données sans valeurs NaN

    Args:
        prefix (str): price_ or product_count_
    """
    columns = [prefix+str(i+1) for i in range(24)]
    df_extracted = df[columns]

    s = df_extracted[columns[0]].squeeze().dropna()
    for product_count_column in columns[1:]:
        s = pd.concat([s, df_extracted[product_count_column].squeeze().dropna()], ignore_index=True)

    return s


def create_submission_csv(y_score, identifiers, filename):
    """Création du fichier csv pour soumission au challenge

    Args:
        y_score (numpy.ndarray):  vecteur des scores de type (n,)
        identifiers (list or pd.Series): liste des id ordonnés comme x_test
        filename (str): nom du fichier sans extension qui sera sauvegardé
                        dans ../data/prediction/filename.csv
    """
    y_pred = pd.DataFrame({'ID': identifiers, 'fraud_flag': y_score.tolist()})
    filename = filename + '.csv'
    y_pred.to_csv(os.path.join('../data/prediction/', filename), index_label='index')
