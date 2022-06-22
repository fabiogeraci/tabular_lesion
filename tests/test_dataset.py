import os.path
import sys
import pandas as pd
# Specific Import for Test
import pytest
from pytest import mark
from confest import path_test_csv, get_dataset

from src.dataset import DataSet


class TestDataset:

    @mark.test_drop_all_zero_columns
    def test_drop_all_zero_columns(self, get_dataset):
        dataset = get_dataset
        assert len(DataSet.drop_all_zero_columns(dataset)) > 0

    @mark.test_drop_columns_std_larger
    def test_drop_columns_std_larger(self, get_dataset):
        dataset = get_dataset
        assert len(DataSet.drop_columns_std_larger(dataset)) > 0
