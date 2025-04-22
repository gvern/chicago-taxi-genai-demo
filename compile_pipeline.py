import kfp.v2
from kfp.v2 import compiler
import os

# Assuming the pipeline definition is in src.pipelines.forecasting_pipeline
# Adjust the import path if necessary based on your project structure and how
# you run this script (e.g., from the root directory).
from src.pipelines.forecasting_pipeline import forecasting_pipeline

# Define the output path for the compiled pipeline JSON
pipeline_package_path = "forecasting_pipeline.json"

# Define the path to the SQL query file *relative to this script*
# This path is used *during compilation* if you choose to embed the SQL content.
# However, based on the current implementation (option 4b), this path isn't
# strictly needed here, but the path *inside the container* is passed to the pipeline function.
# Let's keep the path definition for clarity, assuming it's needed by the caller.
_sql_query_local_path = "src/pipelines/components/preprocessing/bigquery_queries.sql"

# --- Option 4b: Path inside container ---
# Define the path where the SQL file will be located *inside* the Docker image
# This path will be passed as an argument to the pipeline function.
sql_query_path_in_container = "/app/src/pipelines/components/preprocessing/bigquery_queries.sql"

# --- Option 4a: Read SQL Content (Alternative) ---
# If you modify the pipeline to accept SQL content directly (Option 4a):
# try:
#     with open(_sql_query_local_path, 'r') as f:
#         sql_query_content_for_compile = f.read()
#     print(f"Read SQL content from {_sql_query_local_path}")
# except Exception as e:
#     print(f"Error reading SQL file {_sql_query_local_path}: {e}")
#     # Handle error appropriately, maybe exit or use a default query
#     sql_query_content_for_compile = "" # Or raise error


print(f"Compiling pipeline definition to: {pipeline_package_path}")

# Compile the pipeline
# Pass the *path inside the container* as the argument 'sql_query_path'
# If using Option 4a, you would pass sql_query_content=sql_query_content_for_compile instead.
compiler.Compiler().compile(
    pipeline_func=forecasting_pipeline,
    package_path=pipeline_package_path,
    # type_check=True # Optional: Enable type checking during compilation
)

print("Pipeline compilation complete.")
print(f"Pipeline package saved to: {os.path.abspath(pipeline_package_path)}")

