# Specific Import for Test
from pytest import mark

from classifier.dataset import DataSet


@mark.usefixtures("get_dataset")
class TestDataset:

    @mark.test_drop_all_zero_columns
    def test_drop_all_zero_columns(self, get_dataset):
        dataset = get_dataset
        assert len(DataSet.drop_all_zero_columns(dataset)) > 0

    @mark.test_drop_columns_std_larger
    def test_drop_columns_std_larger(self, get_dataset):
        dataset = get_dataset
        assert len(DataSet.drop_columns_std_larger(dataset)) > 0
