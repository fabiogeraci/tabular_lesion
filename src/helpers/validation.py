import os
import pandas as pd
import time

from sklearn.model_selection import cross_validate, ShuffleSplit
from variance import DataVariance

time_stamp = time.strftime("%Y%m%d-%H%M%S")


class ModelValidation:
    def __init__(self, pipe=None, data: DataVariance = None):
        self.pipe = pipe
        self.data = data
        self.cv = ShuffleSplit(n_splits=30, test_size=0.2)
        self.best_estimator_cv_scores()
        self.pipeline_cv_scores()

    def best_estimator_cv_scores(self):

        cv_results = cross_validate(self.pipe.bebest_estimator_, self.data.X_train, self.data.y_train,
                                    cv=self.cv, scoring="neg_mean_absolute_error",
                                    return_train_score=True, n_jobs=2)

        pd.DataFrame.from_dict(cv_results).to_csv(os.path.join('../..', 'data', f'best_estimator_cv_results_{time_stamp}.csv'))

    def pipeline_cv_scores(self):
        pd.DataFrame.from_dict(self.pipe.cv_results_).to_csv(os.path.join('../..', 'data', 'pipe_cv_results_.csv'))





