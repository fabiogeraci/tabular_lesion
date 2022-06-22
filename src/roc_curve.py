import sklearn
import pandas as pd
import time

import matplotlib.pyplot as plt
from variance import DataVariance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc, \
    precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class RocCurve:
    def __init__(self, model: sklearn, a_data: DataVariance = None):
        self.model = model
        self.data = a_data
        self.initialize()

    def initialize(self):
        fpr_lr_train, tpr_lr_train, roc_auc_lr_train = self.generate_score(self.data.X_train, self.data.y_train)
        fpr_lr_test, tpr_lr_test, roc_auc_lr_test = self.generate_score(self.data.X_test, self.data.y_test.values.ravel())

        self.write_columns_to_csv(roc_auc_lr_train, roc_auc_lr_test)

        self.plotly_roc_curve(fpr_lr_train, tpr_lr_train, roc_auc_lr_train, fpr_lr_test, tpr_lr_test, roc_auc_lr_test)

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

    def write_columns_to_csv(self, roc_auc_train: float, roc_auc_test: float):
        """

        """
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        df = pd.DataFrame(self.data.X_train.columns, columns=['Feature'])
        df.to_csv(f'columns_{roc_auc_train:.3f}_{roc_auc_test:.3f}_{time_stamp}.csv', index=False)

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

        plt.show(block=True)

    @staticmethod
    def plotly_roc_curve(fpr_lr_train, tpr_lr_train, roc_auc_lr_train,
                         fpr_lr_test, tpr_lr_test, roc_auc_lr_test):

        roc = {'train': [fpr_lr_train, tpr_lr_train, roc_auc_lr_train],
               'test': [fpr_lr_test, tpr_lr_test, roc_auc_lr_test]}

        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        name = f"ROC Train (AUC={roc['train'][2]:.3f})"
        fig.add_trace(go.Scatter(x=roc['train'][0], y=roc['train'][1], name=name, mode='lines'))

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

        fig.show(block=True)
