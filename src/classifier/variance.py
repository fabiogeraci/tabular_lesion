import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import ADASYN
import sklearn
from feature_select import Selector

warnings.filterwarnings('ignore')


class DataVariance:
    def __init__(self, all_set, variance_flag: bool = False, selector_model: sklearn = None):
        self.all_set = all_set
        self.variance_flag = variance_flag
        self.selector_model = selector_model
        self.X_train_trans = None
        self.X_test_trans = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.variance_mask = None
        self.selector_mask = None
        self.initialize()

    def initialize(self):
        self.variance_mask = self.generate_variance_mask(self.all_set.X_train)
        self.apply_variance()
        self.apply_genetic()
        self.resample_dataset()

    @staticmethod
    def generate_variance_mask(x_set: pd.DataFrame) -> np.array:
        """

        :param x_set:
        :return:
        """
        variance_threshold = VarianceThreshold(threshold=0.005)
        scaler = MaxAbsScaler()
        scaler.fit(x_set)
        _set = scaler.transform(x_set)
        variance_threshold.fit(_set)
        return variance_threshold.get_support()

    def apply_variance(self):
        """

        """
        self.X_train_trans = self.all_set.X_train[self.all_set.X_train.columns[self.variance_mask]]
        self.X_test_trans = self.all_set.X_test[self.all_set.X_test.columns[self.variance_mask]]

    def genetic_feature(self) -> np.array:
        """

        :return:
        """
        model_selector = Selector(self.selector_model)
        return model_selector.selector_model.fit(self.X_train_trans, self.all_set.y_train)

    def apply_genetic(self):
        """

        """
        self.selector_mask = self.genetic_feature()

        self.X_train_trans = self.X_train_trans[self.X_train_trans.columns[self.selector_mask.support_]]
        self.X_test_trans = self.X_test_trans[self.X_test_trans.columns[self.selector_mask.support_]]

    def resample_dataset(self):
        """

        """
        smote = ADASYN(random_state=2022, sampling_strategy='minority', n_jobs=4)
        if self.variance_flag:
            self.X_train, self.y_train = smote.fit_resample(self.X_train_trans, self.all_set.y_train.values.ravel())
            self.X_test = self.X_test_trans
        else:
            self.X_train, self.y_train = smote.fit_resample(self.all_set.X_train, self.all_set.y_train.values.ravel())
            self.X_test = self.all_set.X_test
        self.y_test = self.all_set.y_test
