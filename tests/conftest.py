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
def path_test_csv():
    curr_path = str(pathlib.Path.cwd())
    if 'tests' in curr_path:
        path_test_csv = os.path.join('../', 'data')
    else:
        path_test_csv = os.path.join(curr_path, 'data')
    yield path_test_csv


@pytest.fixture(scope='session')
def get_dataset(path_test_csv):
    """
    Get the dataset
    :param path_test_csv:
    :return:
    """
    return pd.read_csv(os.path.join(path_test_csv, 'lesion_df_balanced_Target_Lesion_ClinSig.csv')).iloc[:, 2:]


@pytest.fixture(scope='session')
def time_stamp():
    # add data stamp to file name
    date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    return date



