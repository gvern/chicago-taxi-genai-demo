# src/pipelines/components/launch_hpt_job.py

from kfp.v2.dsl import component, Output, Model

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform"]
)
def launch_hpt_job(
    project: str,
    location: str,
    staging_bucket: str,
    display_name_prefix: str,
    worker_pool_spec: dict,
    hpt_config: dict,
    static_args: dict,
    training_data_uri: str,
    model_gcs_path: Output[Model]
):
    from google.cloud import aiplatform
    import uuid
    import time

    aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)
    job_id = f"{display_name_prefix}-{uuid.uuid4().hex[:8]}"
    display_name = job_id

    # Prepare command line args for the training script
    args = [
        f"--project_id={project}",
        f"--bq_dataset={training_data_uri.split('.')[1]}",  # crude extraction, adjust as needed
        f"--bq_table={training_data_uri.split('.')[-1]}",
        f"--target_col={static_args['target_column']}",
        f"--id_col={static_args['series_id_column']}",
        f"--time_col={static_args['time_column']}",
        # Add other static args as needed
    ]
    # Optionally add train/val split dates if needed

    # Prepare worker pool spec for CustomJob
    worker_pool = [
        {
            "machine_spec": {
                "machine_type": worker_pool_spec["machine_type"]
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": worker_pool_spec["container_uri"],
                "args": args,
            },
        }
    ]

    # Prepare study spec for HPT
    study_spec = {
        "metrics": [
            {
                "metric_id": hpt_config["metric_tag"],
                "goal": hpt_config["metric_goal"]
            }
        ],
        "parameters": [],
        "algorithm": hpt_config.get("search_algorithm", "RANDOM_SEARCH")
    }
    for param in hpt_config["parameter_spec"].values():
        param_spec = {
            "parameter_id": param["parameter_id"],
            "scale_type": param["scale_type"],
        }
        if param["type"] == "DOUBLE":
            param_spec["double_value_spec"] = {
                "min_value": param["min_value"],
                "max_value": param["max_value"]
            }
        elif param["type"] == "INTEGER":
            param_spec["integer_value_spec"] = {
                "min_value": param["min_value"],
                "max_value": param["max_value"]
            }
        study_spec["parameters"].append(param_spec)

    # Launch the HPT job
    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=display_name,
        custom_job=aiplatform.CustomJob(
            display_name=display_name,
            worker_pool_specs=worker_pool,
            base_output_dir=staging_bucket,
        ),
        metric_spec={hpt_config["metric_tag"]: hpt_config["metric_goal"].lower()},
        parameter_spec={
            param["parameter_id"]: (
                aiplatform.hyperparameter_tuning.DoubleParameterSpec(
                    min=param["min_value"], max=param["max_value"], scale=param["scale_type"].lower()
                ) if param["type"] == "DOUBLE" else
                aiplatform.hyperparameter_tuning.IntegerParameterSpec(
                    min=param["min_value"], max=param["max_value"], scale=param["scale_type"].lower()
                )
            )
            for param in hpt_config["parameter_spec"].values()
        },
        max_trial_count=hpt_config["max_trial_count"],
        parallel_trial_count=hpt_config["parallel_trial_count"],
        search_algorithm=hpt_config.get("search_algorithm", "random_search"),
    )

    hpt_job.run(sync=True)

    # Retrieve best trial's model GCS path
    best_trial = hpt_job.trials[0]
    for trial in hpt_job.trials:
        if trial.final_measurement.metrics[0].value < best_trial.final_measurement.metrics[0].value:
            best_trial = trial
    model_gcs_path_uri = f"{best_trial.output_dir}/model.xgb"
    with open(model_gcs_path.uri, "w") as f:
        f.write(model_gcs_path_uri)