import importlib.util
import os

def test_train_xgboost_hpt_importable():
    script_path = os.path.join(
        os.path.dirname(__file__),
        "../src/model_training/train_xgboost_hpt.py"
    )
    spec = importlib.util.spec_from_file_location("train_xgboost_hpt", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
