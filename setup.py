from setuptools import setup, find_packages

setup(
    name="chicago-taxi-genai-demo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "google-cloud-bigquery>=3.11.0",
        "google-cloud-aiplatform>=1.36.0",
        "kfp>=2.5.0",
        "apache-beam>=2.45.0",
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
        "google-cloud-pipeline-components>=0.1.0",
        "xgboost>=2.0.0",
        "kfp-sdk>=2.5.0",
    ],
) 