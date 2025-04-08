from setuptools import setup, find_packages

setup(
    name="chicago-taxi-forecasting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "google-cloud-bigquery",
        "google-cloud-aiplatform",
        "kfp",
        "pandas",
    ],
) 