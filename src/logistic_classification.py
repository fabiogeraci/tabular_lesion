import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import plotly.express as px
import copy

import sklearn
print(sklearn.__version__)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score, roc_auc_score,  roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, KBinsDiscretizer, Normalizer, PowerTransformer, SplineTransformer, MaxAbsScaler
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


class WorkingSet:
    def __init__(self, csv_file_name: str):
        self.imported_dataframe = pd.read_csv(csv_file_name)
        self.training_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.target_name = ''
        self.generate_train_set()
        self.generate_test_set()

    def generate_train_set(self):
        """
        Generate the training set
        """
        df_train = self.imported_dataframe[self.imported_dataframe['Inf_Train_test'].str.contains('train', case=False)]
        df_valid = self.imported_dataframe[self.imported_dataframe['Inf_Train_test'].str.contains('valid', case=False)]

        self.training_df = pd.concat([df_train, df_valid], axis=0)

        print(f'Are thre any Nan = {self.training_df.isnull().values.any()}, Number of Nan = {self.training_df.isnull().sum().sum()}')

        for key in self.training_df.keys():
            if 'target' in key.lower():
                self.target_name = key
                print(self.target_name)

        self.X_train = self.training_df.drop(['Target_Lesion_ClinSig', 'Inf_Train_test'], axis=1)
        self.X_train = self.drop_all_zero_columns(self.X_train)
        self.X_train = self.drop_columns_std_larger(self.X_train)

        self.y_train = self.training_df['Target_Lesion_ClinSig']

    def generate_test_set(self):
        """
        Generate the test set
        """
        df_test = self.imported_dataframe[self.imported_dataframe['Inf_Train_test'].str.contains('test', case=False)]
        self.X_test = df_test.drop(['Target_Lesion_ClinSig', 'Inf_Train_test'], axis=1)
        self.X_test = self.drop_all_zero_columns(self.X_test)
        self.X_test = self.drop_columns_std_larger(self.X_test)
        self.y_test = df_test['Target_Lesion_ClinSig']

    @staticmethod
    def drop_all_zero_columns(a_dataframe: pd.DataFrame) -> pd.DataFrame:
        return a_dataframe.loc[:, a_dataframe.ne(0).any()]

    @staticmethod
    def drop_columns_std_larger(a_dataframe: pd.DataFrame) -> pd.DataFrame:
        return a_dataframe.loc[:, a_dataframe.std() < 10000]


def variance_threshold(x_set: pd.DataFrame) -> np.array:
    """
    Eleminates features with a variance below 0.005
    :param x_set:
    :return:
    """
    variance_threshold = VarianceThreshold(threshold=0.005)

    transformer = MaxAbsScaler().fit(x_set)
    _set = transformer.transform(x_set)
    variance_threshold.fit(_set)
    variance_threshold.get_support()
    return variance_threshold.transform(_set)


class DataVariance:
    def __init__(self, all_set, variance_flag: bool = False):
        self.all_set = all_set
        self.variance_flag = variance_flag
        self.X_train_trans = None
        self.X_test_trans = None
        self.X_train = None
        self.y_train = None
        self.variance_mask = None
        self.initialize()

    def initialize(self):
        self.variance_mask = self.generate_variance_mask(self.all_set.X_train)
        self.apply_variance()
        self.resample_dataset()

    @staticmethod
    def generate_variance_mask(x_set: pd.DataFrame) -> np.array:
        """
        Generate a mask with the high variance columns
        :param x_set: input features set
        :return: an array with True for high variance and False for low variance features
        """
        variance_threshold = VarianceThreshold(threshold=0.005)
        transformer = MaxAbsScaler().fit(x_set)
        _set = transformer.transform(x_set)
        variance_threshold.fit(_set)
        return variance_threshold.get_support()

    def apply_variance(self):
        """
        Filters out the low variance features
        """
        self.X_train_trans = self.all_set.X_train[self.all_set.X_train.columns[(self.variance_mask)]]
        self.X_test_trans = self.all_set.X_test[self.all_set.X_test.columns[(self.variance_mask)]]

    def resample_dataset(self):
        """
        Over samples the sets
        """
        smote = ADASYN(random_state=2022, sampling_strategy='minority', n_jobs=4)
        if self.variance_flag:
            self.X_train, self.y_train = smote.fit_resample(self.X_train_trans, self.all_set.y_train.values.ravel())
        else:
            self.X_train, self.y_train = smote.fit_resample(self.all_set.X_train.values, self.all_set.y_train.values.ravel())


class RocCurve:
    def __init__(self, a_model: sklearn, a_dataset: WorkingSet, variance: DataVariance = None):
        self.model = a_model
        self.a_dataset = a_dataset
        self.variance = variance
        self.initialize()

    def initialize(self):

        fpr_lr_train, tpr_lr_train, roc_auc_lr_train = self.generate_score(self.variance.X_train, self.variance.y_train)
        fpr_lr_test, tpr_lr_test, roc_auc_lr_test = self.generate_score(self.variance.X_test_trans, self.a_dataset.y_test.values.ravel())

        self.plot_roc_curve(fpr_lr_train, tpr_lr_train, roc_auc_lr_train, fpr_lr_test, tpr_lr_test, roc_auc_lr_test)

    def generate_score(self, x_set, y_set):
        """

        :param x_set:
        :param y_set:
        :return:
        """
        y_scores = self.model.predict(x_set)

        fpr_lr, tpr_lr, _ = roc_curve(y_set, y_scores)
        roc_auc_lr = auc(fpr_lr, tpr_lr)

        return fpr_lr, tpr_lr, roc_auc_lr

    @staticmethod
    def plot_roc_curve(fpr_lr_train, tpr_lr_train, roc_auc_lr_train, fpr_lr_test, tpr_lr_test, roc_auc_lr_test):
        """

        :param fpr_lr_train:
        :param tpr_lr_train:
        :param roc_auc_lr_train:
        :param fpr_lr_test:
        :param tpr_lr_test:
        :param roc_auc_lr_test:
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
        ax1.set_title('ROC curve Train', fontsize=16)
        ax1.plot(fpr_lr_train, tpr_lr_train, lw=3, label=f'LogRegr ROC curve (area = {roc_auc_lr_train:0.2f})')
        ax1.set_xlabel('False Positive Rate', fontsize=16)
        ax1.set_ylabel('True Positive Rate', fontsize=16)
        ax1.legend(loc='lower right', fontsize=13)
        ax1.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

        ax2.set_title('ROC curve Test', fontsize=16)
        ax2.plot(fpr_lr_test, tpr_lr_test, lw=3, label=f'LogRegr ROC curve (area = {roc_auc_lr_test:0.2f})')
        ax2.set_xlabel('False Positive Rate', fontsize=16)
        ax2.set_ylabel('True Positive Rate', fontsize=16)
        ax2.legend(loc='lower right', fontsize=13)
        ax2.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

def plot_class_balance(all_set: WorkingSet):
    df_clinically_Sig = pd.DataFrame(all_set.training_df[all_set.target_name].value_counts())
    df_clinically_Sig.reset_index(inplace=True)
    df_clinically_Sig = df_clinically_Sig.rename(columns={'index': 'Clinically_Sig'})
    df_clinically_Sig = df_clinically_Sig.rename(columns={all_set.target_name: 'Count'})

    fig = px.bar(df_clinically_Sig, x='Clinically_Sig', y='Count', color=('blue', 'red'), text='Count', title='Class Balance')
    fig.update_layout(showlegend=False)
    # fig.show(renderer="colab")


def make_feature_union():
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
    feature_transform = FeatureUnion(transforms)
    return feature_transform


def make_logistic_classifier():
    model = LogisticRegression(random_state=2022,
                            max_iter=100000,
                            penalty='elasticnet',
                            solver='saga',
                            n_jobs=6,
                            warm_start=True,
                            multi_class='auto',
                            tol=1e-4
                            )
    lr_param_grid = {
        'classifier__l1_ratio': [0.2, 0.225, 0.25],
        'classifier__C': [0.0001, 0.0005, 0.001, 0.005, 0.01]
    }

    return model, lr_param_grid


def make_pipeline(a_model, a_feature_transform):

    # define the pipeline
    steps = list()
    steps.append(('scaler', a_feature_transform))
    steps.append(('classifier', a_model))

    return Pipeline(steps=steps)


if __name__ == '__main__':

    all_set = WorkingSet(os.path.join('..', 'data', 'lesion_df_balanced_Target_Lesion_ClinSig.csv'))
    plot_class_balance(all_set)

    assert all_set.X_train.shape[0] == all_set.y_train.shape[0]
    assert all_set.X_test.shape[0] == all_set.y_test.shape[0]

    variance_flag = True
    data = DataVariance(all_set, variance_flag)
    model, lr_param_grid = make_logistic_classifier()

    feature_transform = make_feature_union()

    pipeline = make_pipeline(model, feature_transform)
    LR_search = GridSearchCV(pipeline, param_grid=lr_param_grid, refit=True, verbose=1, cv=10, n_jobs=4)
    LR_search.fit(data.X_train, data.y_train)

    print(LR_search.best_params_)
    # summarize
    print('Mean Accuracy: %.3f' % LR_search.best_score_)
    print('Config: %s' % LR_search.best_params_)

    RocCurve(LR_search.best_estimator_, all_set, data)
