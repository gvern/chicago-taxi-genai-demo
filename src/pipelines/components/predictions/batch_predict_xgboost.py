from kfp.v2.dsl import component, Output, Dataset

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-storage", "pandas", "joblib", "xgboost"]
)
def batch_predict_xgboost(
    model_gcs_path: str,
    features_gcs_path: str,
    output_gcs_path: str,
    predictions: Output[Dataset]
):
    import subprocess
    cmd = [
        "python", "src/xgboost_batch_predict.py",
        "--model_gcs_path", model_gcs_path,
        "--features_gcs_path", features_gcs_path,
        "--output_gcs_path", output_gcs_path,
    ]
    subprocess.run(cmd, check=True)
    predictions.uri = output_gcs_path
