import os

import numpy as np
import sklearn
from skl2onnx import convert_sklearn
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

from classifier.variance import DataVariance


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
