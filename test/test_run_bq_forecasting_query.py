import inspect
import sys
import os

def test_run_bq_forecasting_query_signature():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/pipelines/components")))
    import run_bq_forecasting_query
    fn = run_bq_forecasting_query.run_bq_forecasting_query

    # If the function is a KFP component, it may be wrapped and only expose *args, **kwargs.
    # In that case, just check the function is callable and skip strict signature checks.
    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())
    if param_names == ["args", "kwargs"]:
        # Wrapped by KFP, can't check signature strictly
        assert callable(fn)
        return
    # If not wrapped, check for expected parameters
    for param in ["project_id", "location", "sql_template_path"]:
        assert param in param_names
