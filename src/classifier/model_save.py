import os

import sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType


def save_model(a_model: sklearn, file_name: str):
    """

    :param a_model:
    :param file_name:
    """
    initial_type = [('numfeat', FloatTensorType([None, 3])),
                    ('strfeat', StringTensorType([None, 2]))]
    model_onnx = convert_sklearn(a_model, initial_types=initial_type)

    model_onnx.save(os.path.join('../..', 'models', f'{file_name}.onnx'))
