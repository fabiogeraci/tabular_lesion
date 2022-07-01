# Specific Import for Test
import os.path

import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
import onnxruntime as rt
from onnxmltools.convert import convert_xgboost as convert_xgboost_booster
from skl2onnx.common.data_types import FloatTensorType
from xgboost import XGBClassifier
from pytest import mark


@mark.usefixtures("get_dataset", "path_test_dir")
class TestDataset:

    @mark.test_save_model
    def test_save_model(self, get_dataset, path_test_dir):
        dataset = get_dataset

        _, path_test_model = path_test_dir

        x_train = dataset.iloc[:, 1:]
        y_train = dataset.iloc[:, 0:1]

        pipe = Pipeline([('scaler', StandardScaler()), ('xgb', XGBClassifier(n_estimators=3))])
        pipe.fit(x_train, y_train)

        update_registered_converter(
            XGBClassifier, 'XGBoostXGBClassifier',
            calculate_linear_classifier_output_shapes, convert_xgboost,
            options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

        numbers_features = len(x_train.columns)

        model_onnx = convert_sklearn(
            pipe, 'pipeline_xgboost',
            [('input', FloatTensorType([None, numbers_features]))],
            target_opset={'': 12, 'ai.onnx.ml': 2})

        file_name = os.path.join(path_test_model, "pipeline_xgboost.onnx")

        if os.path.exists(file_name):
            os.remove(file_name)

        with open(file_name, "wb") as f:
            f.write(model_onnx.SerializeToString())

        assert os.path.exists(file_name)


