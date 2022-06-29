import os
from typing import List, Tuple

import numpy as np
import sklearn
from onnxconverter_common import FloatTensorType
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
import onnxruntime as rt
from onnxmltools.convert import convert_xgboost as convert_xgboost_booster
from skl2onnx.common.data_types import FloatTensorType

from variance import DataVariance


class ModelSave:

    def __init__(self, model: sklearn, a_data: DataVariance = None, classifier_name: str = None):

        self.model = model
        self.data = a_data
        self.classifier_name = classifier_name
        self.initial_type: List[Tuple[str, FloatTensorType]] = []
        self.initialize_onnx()

    def initialize_onnx(self):
        number_features = len(self.data.X_train.columns)
        self.initial_type = [('float_input', FloatTensorType([None, number_features]))]


class SklearnModelOnnx:

    @staticmethod
    def save_model(model_save: ModelSave):

        """
        Saves the model as a .onnx file
        """

        # Convert the model to ONNX

        model_onnx = convert_sklearn(model_save.model, initial_types=model_save.initial_type)
        with open(os.path.join('../..', 'models', f'{model_save.classifier_name}.onnx'), "wb") as f:
            f.write(model_onnx.SerializeToString())


class XgboostModelToOnnx(ModelSave):
    def __init__(self, model: XGBClassifier, a_data: DataVariance = None, classifier_name: str = None):
        super().__init__(model, a_data, classifier_name)
        self.save_model()

    def save_model(self):
        update_registered_converter(
            XGBClassifier, 'XGBoostXGBClassifier',
            calculate_linear_classifier_output_shapes, convert_xgboost,
            options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

        model_onnx = convert_sklearn(
            self.model, 'pipeline_xgboost',
            [('input', FloatTensorType([None, self.initial_type]))])

        with open(os.path.join('../..', 'models', f'{self.classifier_name}.onnx'), "wb") as f:
            f.write(model_onnx.SerializeToString())

