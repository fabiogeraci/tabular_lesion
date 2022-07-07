import os
import time

import sklearn

from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsClassifier

from helpers.dataset import DataSet
from helpers.variance import DataVariance
from helpers.model_save import SklearnModelOnnx, ModelSave

from yellowbrick.classifier import ROCAUC

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
        return RidgeClassifier(alpha=0.01,
                               copy_X=True,
                               fit_intercept=True,
                               max_iter=10000,
                               random_state=2022,
                               solver='auto',
                               tol=0.001)

    @staticmethod
    def set_param_grid():
        return {
            'classifier__alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01],
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

    # RocCurve(search, data, 'RidgeClassifier')
    visualizer = ROCAUC(search.best_estimator_, classes=["0", "1"], binary=True)
    visualizer.fit(data.X_train, data.y_train)  # Fit the training data to the visualizer
    visualizer.score(data.X_test, data.y_test)  # Evaluate the model on the test data

    visualizer.show(outpath=f"../../images/RidgeClassifier_{time_stamp}.png")  # Finalize and render the figure

    model_save = ModelSave(search.best_estimator_, data, f'RidgeClassifier_{time_stamp}')

    SklearnModelOnnx.save_model(model_save)
