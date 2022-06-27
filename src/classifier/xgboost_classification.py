import os

import sklearn
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, SGDClassifier, SGDOneClassSVM
from sklearn.pipeline import Pipeline, FeatureUnion

from dataset import DataSet
from variance import DataVariance
from roc_curve import RocCurve
from plotters.plot_class_balance import plot_class_balance

import warnings
warnings.filterwarnings('ignore')
print(sklearn.__version__)


class Model:
    def __init__(self):
        self.classifier = None
        self.param_grid = None
        self.make_classifier()

    def make_classifier(self):
        """

        :return:
        """
        self.classifier = XGBClassifier(random_state=2022,
                                        booster='gbtree',
                                        # objective='binary:logistic',
                                        n_estimators=100,
                                        eta=0.001)
        self.param_grid = {
            # 'classifier__booster': ['gbtree', 'gblinear', 'dart'],
            # 'classifier__eta': [0.1, 0.2, 0.3],
            'classifier__gamma': [0.00025, 0.0005, 0.001, 0.002],
            'classifier__max_depth': [5, 6, 7, 8],
        }


def make_feature_union():
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
    transform_feature = FeatureUnion(transforms)
    return transform_feature


def make_pipeline(a_model, a_feature_transform):
    """

    :param a_model:
    :param a_feature_transform:
    :return:
    """
    # define the pipeline
    steps = list()
    steps.append(('scaler', a_feature_transform))
    steps.append(('classifier', a_model))

    return Pipeline(steps=steps)


if __name__ == '__main__':
    all_set = DataSet(os.path.join('../..', 'data', 'lesion_df_balanced_Target_Lesion_ClinSig.csv'))
    # plot_class_balance(all_set)

    assert all_set.X_train.shape[0] == all_set.y_train.shape[0]
    assert all_set.X_test.shape[0] == all_set.y_test.shape[0]

    variance_flag = True
    data = DataVariance(all_set, variance_flag, KNeighborsClassifier(n_neighbors=25))

    model = Model()

    feature_transform = make_feature_union()

    pipeline = make_pipeline(model.classifier, feature_transform)
    search = GridSearchCV(pipeline, param_grid=model.param_grid, scoring='f1_weighted', refit=True, verbose=1, cv=10, n_jobs=4)
    search.fit(data.X_train, data.y_train)

    print(search.best_params_)
    # summarize
    print('Mean Accuracy: %.3f' % search.best_score_)
    print('Config: %s' % search.best_params_)

    RocCurve(search, data, 'XGBClassifier_KNC25f_binary_logistic')
