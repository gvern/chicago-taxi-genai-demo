from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas as pd

# === PARAMÈTRES ===
PROJECT_ID = "avisia-certification-ml-yde"
SOURCE_TABLE = "avisia-certification-ml-yde.chicago_taxis.taxi_trips"
DEST_DATASET = "chicago_taxis"
DEST_TABLE = "taxi_trips"
DEST_REGION = "europe-west1"
CHUNK_SIZE = 100_000  # nombre de lignes par batch
START_FROM_BATCH = 1509  # Reprendre à partir de ce batch (le dernier échoué)

# === Clients ===
bq_us = bigquery.Client(project=PROJECT_ID, location="US")
bq_eu = bigquery.Client(project=PROJECT_ID, location=DEST_REGION)

# === Création du dataset s'il n'existe pas ===
dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{DEST_DATASET}")
dataset_ref.location = DEST_REGION

try:
    bq_eu.get_dataset(dataset_ref)
    print(f"✅ Dataset déjà présent : {DEST_DATASET}")
except NotFound:
    print(f"📦 Création du dataset {DEST_DATASET} dans {DEST_REGION}")
    bq_eu.create_dataset(dataset_ref)

# === Vérifier si la table existe déjà ===
table_exists = False
try:
    bq_eu.get_table(f"{PROJECT_ID}.{DEST_DATASET}.{DEST_TABLE}")
    table_exists = True
    print(f"✅ Table {DEST_TABLE} existe déjà, on va continuer avec WRITE_APPEND")
except NotFound:
    table_exists = False
    print(f"⚠️ Table {DEST_TABLE} n'existe pas, elle sera créée")

# === Construction de la requête source ===
query = f"SELECT * FROM `{SOURCE_TABLE}` LIMIT {CHUNK_SIZE * (START_FROM_BATCH + 100)}"
if START_FROM_BATCH > 1:
    print(f"🔄 Reprise à partir du batch {START_FROM_BATCH}")
    # Calculer l'offset pour reprendre au bon endroit
    offset = (START_FROM_BATCH - 1) * CHUNK_SIZE
    query = f"SELECT * FROM `{SOURCE_TABLE}` LIMIT {CHUNK_SIZE * 100} OFFSET {offset}"

# === Exécution de la requête en batchs ===
print("🚚 Début du streaming batch BigQuery US → EU...")

query_job = bq_us.query(query)
iterator = query_job.result(page_size=CHUNK_SIZE)
pages = iterator.pages

batch_num = 0
for page in pages:
    # Convert the rows in the current page to a DataFrame
    rows = list(page)
    if not rows:
        print(f"📦 Batch {batch_num + 1} - empty page, skipping.")
        continue # Skip empty pages

    # Construct DataFrame from rows
    df_batch = pd.DataFrame(
        data=[dict(row.items()) for row in rows],
        columns=[field.name for field in iterator.schema] # Use schema from the iterator
    )
    batch_num += 1

    print(f"📦 Batch {batch_num} → {len(df_batch)} lignes")

    job_config = bigquery.LoadJobConfig(
        write_disposition=(
            "WRITE_TRUNCATE" if batch_num == 1 else "WRITE_APPEND"
        ),
        autodetect=True,
    )

    job = bq_eu.load_table_from_dataframe(
        df_batch,
        destination=f"{PROJECT_ID}.{DEST_DATASET}.{DEST_TABLE}",
        job_config=job_config,
    )
    job.result()
    print(f"✅ Batch {batch_num} inséré")

print("🎉 Importation complète dans BigQuery Europe.")
