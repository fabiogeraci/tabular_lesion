import warnings
import os
import numpy as np
import pandas as pd
import plotly.express as px
import copy
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, VarianceThreshold
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SMOTEN, SMOTENC, SVMSMOTE
import sklearn
warnings.filterwarnings('ignore')
print(sklearn.__version__)


class DataVariance:
    def __init__(self, all_set, variance_flag: bool = False):
        self.all_set = all_set
        self.variance_flag = variance_flag
        self.X_train_trans = None
        self.X_test_trans = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.variance_mask = None
        self.initialize()

    def initialize(self):
        self.variance_mask = self.generate_variance_mask(self.all_set.X_train)
        self.apply_variance()
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
        self.X_train_trans = self.all_set.X_train[self.all_set.X_train.columns[(self.variance_mask)]]
        self.X_test_trans = self.all_set.X_test[self.all_set.X_test.columns[(self.variance_mask)]]

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
