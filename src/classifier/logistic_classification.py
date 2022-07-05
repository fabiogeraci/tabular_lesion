import os
import time

import sklearn
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from helpers.dataset import DataSet
from helpers.variance import DataVariance
from helpers.roc_curve import RocCurve
from helpers.model_save import SklearnModelOnnx, ModelSave

# from plotters.plot_class_balance import plot_class_balance

import warnings
warnings.filterwarnings('ignore')
print(sklearn.__version__)
time_stamp = time.strftime("%Y%m%d-%H%M%S")


class Model:
    def __init__(self):
        self.classifier = None
        self.param_grid = None
        self.make_classifier()

    def make_classifier(self):
        """

        :return:
        """
        self.classifier = LogisticRegression(random_state=2022,
                                             max_iter=100000,
                                             penalty='elasticnet',
                                             solver='saga',
                                             n_jobs=6,
                                             warm_start=True,
                                             multi_class='auto',
                                             tol=1e-4
                                             )
        self.param_grid = {
            'classifier__l1_ratio': [0.2, 0.225, 0.25],
            'classifier__C': [0.0001, 0.0005, 0.001, 0.005, 0.01]
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
    search = GridSearchCV(pipeline, param_grid=model.param_grid, refit=True, verbose=1, cv=10, n_jobs=4)
    search.fit(data.X_train, data.y_train)

    print(search.best_params_)
    # summarize
    print('Mean Accuracy: %.3f' % search.best_score_)
    print('Config: %s' % search.best_params_)

    roc_curve_data = RocCurve(search, data, 'LogisticRegression')

    model_save = ModelSave(search.best_params_, data, f'LogisticRegression_{time_stamp}')

    SklearnModelOnnx.save_model(model_save)
