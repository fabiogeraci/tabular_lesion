# Test related Imports
import pytest
import os
import pathlib
import datetime
import gc

# Specific Functional Imports
import pandas as pd

# Garbage collect
gc.collect()

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)


@pytest.fixture(scope='session')
def path_test_dir():
    curr_path = str(pathlib.Path.cwd())
    if 'tests' in curr_path:
        path_test_csv = os.path.join('../', 'data')
        path_test_model = os.path.join('../', 'models')
    else:
        path_test_csv = os.path.join(curr_path, 'data')
        path_test_model = os.path.join(curr_path, 'models')
    yield path_test_csv, path_test_model


@pytest.fixture(scope='session')
def get_dataset(path_test_dir):
    """
    Get the dataset
    :param path_test_dir:
    :return:
    """
    path_test_csv, _ = path_test_dir
    return pd.read_csv(os.path.join(path_test_csv, 'lesion_df_balanced_Target_Lesion_ClinSig.csv')).iloc[:, 1:]


@pytest.fixture(scope='session')
def time_stamp():
    # add data stamp to file name
    date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    return date



