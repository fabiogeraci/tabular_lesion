# Test related Imports
import pytest
import csv
import os
import glob
import pathlib
import datetime
import gc
import sys


# Garbage collect
gc.collect()

# Specific Functional Imports
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)


@pytest.fixture(scope='session')
def instance_path():
    curr_path = str(pathlib.Path.cwd())
    if 'tests' in curr_path:
        path_test_csv = os.path.join(curr_path, 'cases')
    else:
        path_test_csv = os.path.join(curr_path, 'tests/cases')
    yield path_test_csv


@pytest.fixture(scope='session')
def time_stamp():
    # add data stamp to file name
    date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    return date



