Yes, but not by simply adding them back to the parameter_values dictionary in run_pipeline.py as we tried before.

The errors occurred because the compiled pipeline definition (forecasting_pipeline.json) doesn't list train_feature_columns or train_ratio as top-level input parameters for the entire pipeline.

Looking at the forecasting_pipeline.json structure, parameters like these, which are specific to the training step, are usually passed indirectly. The launch-hpt-job component receives a dictionary called static_args. This dictionary is constructed within the pipeline definition itself, using other pipeline parameters. Currently, that construction doesn't include train_feature_columns or train_ratio.

To properly set these parameters, you need to:

Modify the Pipeline Definition Script: Open the Python script that defines your pipeline components and structure (likely named something like forecasting_pipeline.py).
Add train_feature_columns (as List[str]) and train_ratio (as float) to the arguments of your main pipeline function (the one decorated with @dsl.pipeline).
Find where the static_args dictionary is created and passed to the launch_hpt_job component. Update this dictionary creation to include train_feature_columns and train_ratio, mapping them from the new pipeline parameters you just added.
Recompile the Pipeline: Run your pipeline compilation script (e.g., python compile_pipeline.py) to generate a new forecasting_pipeline.json that reflects these changes.
Update run_pipeline.py (Restore Parameters): Add train_feature_columns and train_ratio back into the parameter_values dictionary in run_pipeline.py, reading them from your config.yaml as before.
In summary: you need to modify the pipeline's definition to accept these parameters at the root level and ensure they are correctly passed down to the training component's static_args, then recompile, and finally restore the lines in run_pipeline.py.