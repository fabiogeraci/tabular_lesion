import sklearn
import matplotlib.pyplot as plt
from variance import DataVariance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc, \
    precision_recall_curve


class RocCurve:
    def __init__(self, model: sklearn, a_data: DataVariance = None, a_scaler=None):
        self.model = model
        self.scaler = a_scaler
        self.data = a_data
        self.initialize()

    def initialize(self):
        fpr_lr_train, tpr_lr_train, roc_auc_lr_train = self.generate_score(self.data.X_train, self.data.y_train)
        fpr_lr_test, tpr_lr_test, roc_auc_lr_test = self.generate_score(self.data.X_test, self.data.y_test.values.ravel())

        self.plot_roc_curve(fpr_lr_train, tpr_lr_train, roc_auc_lr_train, fpr_lr_test, tpr_lr_test, roc_auc_lr_test)

    def generate_score(self, x_set, y_set):
        y_scores = self.model.predict(self.scaler.transform(x_set))

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
