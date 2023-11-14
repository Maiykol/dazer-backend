import os
import pandas as pd
import random
import string
import numpy as np


ATTEMPTS = 300


def generate_id(size):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

def get_session_files_folder(session):
    return os.path.join("_sessions", session, 'files')

def get_session_subsample_folder(session):
    return os.path.join("_sessions", session, 'subsamples')

def get_session_subsample_task_folder(session, filename, subsample_id):
    return os.path.join("_sessions", session, 'subsamples', filename, subsample_id)

def get_session_subsample_test_folder(session, filename, subsample_id):
    return os.path.join("_sessions", session, 'subsamples', filename, subsample_id, 'test')

def get_session_subsample_train_folder(session, filename, subsample_id):
    return os.path.join("_sessions", session, 'subsamples', filename, subsample_id, 'train')

def get_model_folder(session, filename, classification_task_id):
    return os.path.join("_sessions", session, 'models', filename, classification_task_id, 'train')

def read_file(filename):
    df = pd.read_csv(filename, index_col=0, sep='\t')
    return df

def write_file(df, filename):
    df.to_csv(filename, sep='\t')
    return

def clean_input_dataframe(df):
    rows_removed = 0
    length = len(df.index)
    # drop rows containing NA
    df = df.dropna()
    if length > len(df.index):
        rows_removed = length - len(df.index)
    return df, rows_removed

def get_param_from_filename(target, filename, sep=';'):
    filename = filename.split(os.sep)[-1]
    # remove file ending
    filename = filename[:filename.rfind('.')]
    for param in filename.split(sep):
        if '=' not in param:
            continue
        key, value = param.split('=')
        if key == target:
            return value
    return ''

def get_df_column_information(df):
    columns = list(df.columns)
    categorical_columns = df.columns[~df.columns.isin(list(df.select_dtypes(include=[np.number]).columns.values))]
    categorical_columns_values = {}
    for col in categorical_columns:
        categorical_columns_values[col] = df[col].dropna().unique().tolist()
    return columns, categorical_columns_values
