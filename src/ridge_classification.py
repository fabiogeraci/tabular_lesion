import os
import numpy as np
import pandas as pd
import plotly.express as px
import copy

import sklearn

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc, \
    precision_recall_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, KBinsDiscretizer, Normalizer, PowerTransformer, \
    SplineTransformer, MaxAbsScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, SGDClassifier, SGDOneClassSVM
from sklearn.pipeline import Pipeline, FeatureUnion
from xgboost import XGBRegressor, plot_importance
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SMOTEN, SMOTENC, SVMSMOTE
from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt
from dataset import DataSet
from variance import DataVariance
from feature_select import Selector
from roc_curve import RocCurve

import warnings
warnings.filterwarnings('ignore')
print(sklearn.__version__)


def plot_class_balance(a_dataset: DataSet):
    """
    
    :param a_dataset:
    """
    target_series = pd.DataFrame(a_dataset.training_df[a_dataset.target_name].value_counts())
    target_series.reset_index(inplace=True)
    target_series = target_series.rename(columns={'index': 'Clinically_Sig'})
    target_series = target_series.rename(columns={a_dataset.target_name: 'Count'})

    fig = px.bar(target_series, x='Clinically_Sig', y='Count', color=('blue', 'red'), text='Count', title='Class Balance',
                 width=800, height=400)
    fig.update_layout(showlegend=False)
    # fig.show(renderer="colab")


class Model:
    def __init__(self):
        self.classifier = None
        self.param_grid = None

    def make_logistic_classifier(self):
        """

        :return:
        """
        self.classifier = RidgeClassifier(alpha=0.01,
                                          copy_X=True,
                                          fit_intercept=True,
                                          max_iter=10000,
                                          random_state=2022,
                                          solver='auto',
                                          tol=0.001)
        self.param_grid = {
            'classifier__alpha': [0.001, 0.005, 0.01],
        }


def make_feature_union():
    """

    :return:
    """
    # transforms for the feature union
    transforms = list()
    transforms.append(('maxbbs', MaxAbsScaler()))
    transforms.append(('mms', MinMaxScaler()))
    transforms.append(('ss', StandardScaler()))
    transforms.append(('rs', RobustScaler()))
    transforms.append(('qt', QuantileTransformer(n_quantiles=100, output_distribution='normal')))
    transforms.append(('norm', Normalizer()))
    transforms.append(('pt', PowerTransformer()))
    transforms.append(('st', SplineTransformer()))
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
    all_set = DataSet(os.path.join('..', 'data', 'lesion_df_balanced_Target_Lesion_ClinSig.csv'))
    plot_class_balance(all_set)

    assert all_set.X_train.shape[0] == all_set.y_train.shape[0]
    assert all_set.X_test.shape[0] == all_set.y_test.shape[0]

    variance_flag = True
    data = DataVariance(all_set, variance_flag)

    log_reg = LogisticRegression()
    model_selector = Selector(log_reg)
    model_selector.selector_model.fit(data.X_train, data.y_train)

    model = Model()

    feature_transform = make_feature_union()

    pipeline = make_pipeline(model.classifier, feature_transform)
    LR_search = GridSearchCV(pipeline, param_grid=model.param_grid, refit=True, verbose=1, cv=10, n_jobs=4)
    LR_search.fit(data.X_train, data.y_train)

    print(LR_search.best_params_)
    # summarize
    print('Mean Accuracy: %.3f' % LR_search.best_score_)
    print('Config: %s' % LR_search.best_params_)

    scaler = pipeline['scaler']
    RocCurve(LR_search.best_estimator_, data, scaler)
