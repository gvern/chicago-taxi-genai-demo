import os

def test_bigquery_sql_template_load():
    sql_path = os.path.join(
        os.path.dirname(__file__),
        "../src/data_preprocessing/bigquery_queries.sql"
    )
    with open(sql_path, "r") as f:
        sql = f.read()
    assert "CREATE OR REPLACE TABLE" in sql
    assert "{{ project_id }}" in sql
    assert "{{ dataset_id }}" in sql
    assert "{{ destination_table }}" in sql
