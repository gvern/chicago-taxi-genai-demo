import pandas as pd
from google.cloud import bigquery
from kfp.v2.dsl import component, Input, Output, Dataset, Metrics, HTML
from src.visualization import plot_prediction_vs_actual

@component(
    base_image="python:3.9",  # Or your custom image with required libs
    packages_to_install=["pandas", "google-cloud-bigquery", "matplotlib", "scikit-learn"]
)
def evaluate_visualize_predictions(
    predictions_gcs_path: str,
    actuals_bq_table_uri: str,
    output_dir: str,
    metrics: Output[Metrics],
    html_artifact: Output[HTML]
):
    """
    Compare predictions to actuals, compute metrics, and generate plots.
    """
    # Load predictions
    df_pred = pd.read_csv(predictions_gcs_path)
    # Load actuals from BQ
    client = bigquery.Client()
    df_actual = client.query(f"SELECT * FROM `{actuals_bq_table_uri}`").to_dataframe()
    # Merge on timestamp and area
    df = pd.merge(df_pred, df_actual, on=['timestamp', 'pickup_community_area'], suffixes=('_pred', '_actual'))
    # Compute metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(df['target_actual'], df['prediction'])
    rmse = mean_squared_error(df['target_actual'], df['prediction'], squared=False)
    mape = (abs((df['target_actual'] - df['prediction']) / df['target_actual'])).mean()
    metrics.log_metric("mae", mae)
    metrics.log_metric("rmse", rmse)
    metrics.log_metric("mape", mape)
    # Generate plot
    fig = plot_prediction_vs_actual(df)
    fig_path = f"{output_dir}/prediction_vs_actual.png"
    fig.savefig(fig_path)
    # Save HTML artifact
    with open(html_artifact.path, "w") as f:
        f.write(f"<h2>Forecast Evaluation</h2><ul><li>MAE: {mae:.2f}</li><li>RMSE: {rmse:.2f}</li><li>MAPE: {mape:.2%}</li></ul>")
        f.write(f'<img src="{fig_path}" alt="Prediction vs Actual">')
