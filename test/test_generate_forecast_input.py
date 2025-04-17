import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.pipelines.components.generate_forecast_input import generate_forecast_input

class DummyOutput:
    def __init__(self):
        self.uri = None

@patch("src.pipelines.components.generate_forecast_input.bigquery.Client")
@patch("src.pipelines.components.generate_forecast_input.engineer_temporal_features")
def test_generate_forecast_input(mock_engineer, mock_bq_client, tmp_path):
    mock_client = MagicMock()
    mock_bq_client.return_value = mock_client
    mock_client.query.return_value.to_dataframe.return_value = pd.DataFrame({"pickup_community_area": [1, 2]})
    mock_engineer.side_effect = lambda df, col, period: df

    project_id = "test_project"
    bq_dataset = "test_dataset"
    bq_table_prepared = "test_table"
    id_col = "pickup_community_area"
    time_col = "trip_start_timestamp_hour"
    forecast_horizon_hours = 2
    forecast_start_time = "2024-01-01T00:00:00Z"
    output_gcs_path = str(tmp_path / "future_features.csv")
    future_features = DummyOutput()

    # Call the component function
    generate_forecast_input(
        project_id=project_id,
        bq_dataset=bq_dataset,
        bq_table_prepared=bq_table_prepared,
        id_col=id_col,
        time_col=time_col,
        forecast_horizon_hours=forecast_horizon_hours,
        forecast_start_time=forecast_start_time,
        output_gcs_path=output_gcs_path,
        future_features=future_features
    )

    df = pd.read_csv(output_gcs_path)
    assert not df.empty
    assert future_features.uri == output_gcs_path
