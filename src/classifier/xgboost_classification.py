import os

import pandas as pd
import sklearn
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline, FeatureUnion

from helpers.dataset import DataSet
from helpers.variance import DataVariance
from helpers.roc_curve import RocCurve
from helpers.validation import ModelValidation
from helpers.model_save import XgboostModelToOnnx

# from plotters.plot_class_balance import plot_class_balance

import warnings

warnings.filterwarnings('ignore')
print(sklearn.__version__)


class Model:
    def __init__(self):
        self.classifier = self.initiate_classifier()
        self.param_grid = self.set_param_grid()
        self.feature_transform = self.feature_union
        self.pipeline = self.make_pipeline()

    @staticmethod
    def initiate_classifier():
        """

        :return:
        """
        return XGBClassifier(random_state=2022,
                             objective='binary:logistic',
                             booster='gbtree',
                             n_estimators=100,
                             # colsample_bytree=0.5,
                             # subsample=0.5,
                             # eta=0.001
                             )

    @staticmethod
    def set_param_grid():
        return {
            'classifier__booster': ['gblinear', 'dart', 'gbtree'],
            'classifier__eta': [0.0001, 0.001, 0.1],
            'classifier__n_estimators': [50, 75, 100, 125, 150],
            'classifier__colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6],
            'classifier__subsample': [0.2, 0.3, 0.4, 0.5, 0.6],
            'classifier__gamma': [0.00025, 0.0005, 0.001],
            'classifier__max_depth': [5, 6, 7, 8]
        }

    @property
    def feature_union(self):
        """

        :return:
        """
        # transforms for the feature union
        transforms = list()
        transforms.append(('maxbbs', MaxAbsScaler()))
        # transforms.append(('mms', MinMaxScaler()))
        # transforms.append(('ss', StandardScaler()))
        # transforms.append(('rs', RobustScaler()))
        # transforms.append(('qt', QuantileTransformer(n_quantiles=100, output_distribution='normal')))
        # transforms.append(('norm', Normalizer()))
        # transforms.append(('pt', PowerTransformer()))
        # transforms.append(('st', SplineTransformer()))
        # transform_feature = FeatureUnion(transforms)
        return FeatureUnion(transforms)

    def make_pipeline(self):
        """

        :return:
        """
        # define the pipeline
        steps = list()
        steps.append(('scaler', self.feature_transform))
        steps.append(('classifier', self.classifier))

        return Pipeline(steps=steps)


if __name__ == '__main__':
    all_set = DataSet(os.path.join('../..', 'data', 'lesion_df_balanced_Target_Lesion_ClinSig.csv'))
    # plot_class_balance(all_set)

    assert all_set.X_train.shape[0] == all_set.y_train.shape[0]
    assert all_set.X_test.shape[0] == all_set.y_test.shape[0]

    data = DataVariance(all_set)  # , True, KNeighborsClassifier(n_neighbors=25))

    model = Model()

    search = GridSearchCV(model.pipeline, param_grid=model.param_grid, scoring='f1_weighted',
                          refit=True, verbose=1, cv=10, n_jobs=4)

    search.fit(data.X_train, data.y_train)

    # validation best estimator and pipeline
    ModelValidation(search, data)

    print(search.best_params_)
    # summarize
    print('Mean Accuracy: %.3f' % search.best_score_)
    print('Config: %s' % search.best_params_)

    onnx_file_name = 'XGBClassifier_KNeighborsClassifier'

    roc_curve_data = RocCurve(search, data, onnx_file_name)

    XgboostModelToOnnx(search.best_estimator_, data, f'{onnx_file_name}_{roc_curve_data.test_scores}')
