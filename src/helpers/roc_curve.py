import sklearn
import pandas as pd
import time

from helpers.variance import DataVariance
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

time_stamp = time.strftime("%Y%m%d-%H%M%S")


class RocCurve:
    def __init__(self, model: sklearn, a_data: DataVariance = None, classifier_name: str = None):
        self.model = model
        self.data = a_data
        self.classifier_name = classifier_name
        self.test_scores: str = ''
        self.initialize()

    def initialize(self):
        # fpr_lr_train, tpr_lr_train, roc_auc_lr_train = self.generate_score(self.data.X_train, self.data.y_train)
        fpr_lr_test, tpr_lr_test, roc_auc_lr_test = self.generate_score(self.data.X_test, self.data.y_test.values.ravel())

        self.write_columns_to_csv(roc_auc_lr_test)

        self.plotly_roc_curve(fpr_lr_test, tpr_lr_test, roc_auc_lr_test, self.classifier_name)

    def generate_score(self, x_set, y_set):
        """
        Calculate the FP and TP rate for the ROC analysis
        :param x_set:
        :param y_set:
        :return:
        """
        y_scores = self.model.predict_proba(x_set)

        fpr_lr, tpr_lr, _ = roc_curve(y_set, y_scores[:, 1])
        roc_auc_lr = auc(fpr_lr, tpr_lr)

        return fpr_lr, tpr_lr, roc_auc_lr

    def write_columns_to_csv(self, roc_auc_test: float):
        """
        Writes out the Selected features in csv format

        :param roc_auc_test: ROC accuracy for Test set
        """

        df = pd.DataFrame(self.data.X_train.columns, columns=['Feature'])
        self.test_scores = f'{roc_auc_test:.3f}'
        df.to_csv(f'../../results/{self.classifier_name}_selected_features_{self.test_scores}_{time_stamp}.csv', index=False)

    @staticmethod
    def plotly_roc_curve(fpr_lr_test: float, tpr_lr_test: float, roc_auc_lr_test: float, classifier_name: str):
        """
        Plots and save the plot as a png of the combined ROC curve Train/Test
        :param fpr_lr_test:
        :param tpr_lr_test:
        :param roc_auc_lr_test:
        :param classifier_name:
        """
        roc = {'test': [fpr_lr_test, tpr_lr_test, roc_auc_lr_test]}

        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        # name = f"ROC Train (AUC={roc['train'][2]:.3f})"
        # fig.add_trace(go.Scatter(x=roc['train'][0], y=roc['train'][1], name=name, mode='lines'))

        name = f"ROC Test (AUC={roc['test'][2]:.3f})"
        fig.add_trace(go.Scatter(x=roc['test'][0], y=roc['test'][1], name=name, mode='lines'))

        fig.update_layout(
            uniformtext_minsize=12,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
        fig.write_image(f"../../images/{classifier_name}_ROC_{roc['test'][2]:.3f}_{time_stamp}.png")
        # fig.show(block=True)
