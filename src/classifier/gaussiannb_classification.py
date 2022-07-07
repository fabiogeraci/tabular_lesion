import os
import time

import sklearn
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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
        self.classifier = self.initiate_classifier()
        self.param_grid = self.set_param_grid()
        self.feature_transform = self.feature_union
        self.pipeline = self.make_pipeline()

    @staticmethod
    def initiate_classifier():
        """

        :return:
        """
        return GaussianNB()

    @staticmethod
    def set_param_grid():
        return {
            'classifier__var_smoothing': [1e-11, 1e-10, 1e-9]
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

    variance_flag = True
    data = DataVariance(all_set, variance_flag, KNeighborsClassifier(n_neighbors=25))

    model = Model()

    search = GridSearchCV(model.pipeline, param_grid=model.param_grid, refit=True, verbose=1, cv=10, n_jobs=4)
    search.fit(data.X_train, data.y_train)

    print(search.best_params_)
    # summarize
    print('Mean Accuracy: %.3f' % search.best_score_)
    print('Config: %s' % search.best_params_)

    roc_curve_data = RocCurve(search, data, 'GaussianNB')

    model_save = ModelSave(search.best_estimator_, data, f'GaussianNB_{time_stamp}')

    SklearnModelOnnx.save_model(model_save)
