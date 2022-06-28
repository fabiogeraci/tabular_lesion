import os

import numpy as np
import sklearn
from xgboost import XGBClassifier
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
import onnxruntime as rt
from onnxmltools.convert import convert_xgboost as convert_xgboost_booster
from skl2onnx.common.data_types import FloatTensorType

from classifier.variance import DataVariance
from roc_curve import RocCurve


class ModelSave:
    def __init__(self, model: sklearn, a_data: DataVariance = None, classifier_name: str = None):

        self.model = model
        self.data = a_data
        self.classifier_name = classifier_name
        self.initial_type = None
        self.initialize_onnx()

    def initialize_onnx(self):
        number_features = len(self.data.X_train.columns)
        self.initial_type = [('float_input', FloatTensorType([None, number_features]))]


class SklearnModelOnnx(ModelSave):
    @classmethod
    def save_model(cls, model: sklearn, classifier_name: str = None):
        """
        Saves the model as a .onnx file
        :param model:
        :param classifier_name:
        """
        model_onnx = convert_sklearn(model, initial_types=ModelSave.initialize_onnx)
        with open(os.path.join('../..', 'models', f'{classifier_name}.onnx'), "wb") as f:
            f.write(model_onnx.SerializeToString())


class XgboostModelToOnnx(ModelSave):
    @classmethod
    def save_model(cls, model: sklearn, classifier_name: str = None):
        """
        Saves xgboost model as a .onnx file
        :param model:
        :param classifier_name:
        """

        update_registered_converter(
            XGBClassifier, 'XGBoostXGBClassifier',
            calculate_linear_classifier_output_shapes, convert_xgboost,
            options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

        model_onnx = convert_sklearn(
            model, 'pipeline_xgboost',
            [('input', FloatTensorType([None, ModelSave.initialize_onnx]))])

        with open(os.path.join('../..', 'models', f'{classifier_name}.onnx'), "wb") as f:
            f.write(model_onnx.SerializeToString())


def save_model(a_data: DataVariance, a_model: sklearn, file_name: str):
    """

    :param a_data:
    :param a_model:
    :param file_name:
    """
    number_features = len(a_data.X_train.columns)
    initial_type = [('float_input', FloatTensorType([None, number_features]))]
    model_onnx = convert_sklearn(a_model, initial_types=initial_type)

    with open(os.path.join('../..', 'models', f'{file_name}.onnx'), "wb") as f:
        f.write(model_onnx.SerializeToString())
