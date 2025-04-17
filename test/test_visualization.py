import sys
import os

def test_visualization_functions_exist():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
    import visualization
    assert hasattr(visualization, "plot_time_series_plotly")
    assert hasattr(visualization, "plot_hourly_pattern")
    assert hasattr(visualization, "plot_daily_heatmap")
    assert hasattr(visualization, "plot_top_zones")
    assert hasattr(visualization, "plot_prediction_vs_actual")
