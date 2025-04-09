from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas as pd
from tqdm import tqdm  # Importer tqdm
import math

# === PARAMÈTRES ===
PROJECT_ID = "avisia-certification-ml-yde"
SOURCE_TABLE = "bigquery-public-data.chicago_taxi_trips.taxi_trips"
DEST_DATASET = "chicago_taxis"
DEST_TABLE = "taxi_trips"
DEST_REGION = "europe-west1"
CHUNK_SIZE = 100_000  # nombre de lignes par batch
# START_FROM_BATCH = 1509  # Reprendre à partir de ce batch (le dernier échoué)
START_FROM_BATCH = 1
MAX_BATCHES = 100 # Pour limiter le nombre de batches pour le test


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

# === Vérifier si la table existe déjà et obtenir le nombre de lignes ===
table_exists = False
start_row = 0
table_ref = bq_eu.dataset(DEST_DATASET).table(DEST_TABLE)
try:
    table_info = bq_eu.get_table(table_ref)
    table_exists = True
    start_row = table_info.num_rows
    print(f"✅ Table {DEST_TABLE} existe déjà avec {start_row} lignes, on va continuer avec WRITE_APPEND")
    START_FROM_BATCH = math.ceil(start_row / CHUNK_SIZE) + 1
    print(f"🔄 Reprise automatique à partir du batch {START_FROM_BATCH} (basé sur les lignes existantes)")
except NotFound:
    table_exists = False
    print(f"⚠️ Table {DEST_TABLE} n'existe pas, elle sera créée au premier batch")

# === Construction de la requête source ===
limit_rows = CHUNK_SIZE * MAX_BATCHES
offset = (START_FROM_BATCH - 1) * CHUNK_SIZE
query = f"SELECT * FROM `{SOURCE_TABLE}` LIMIT {limit_rows} OFFSET {offset}"

print(f"Executing query: {query}")

# === Exécution de la requête en batchs ===
print("🚚 Début du streaming batch BigQuery US → EU...")

query_job = bq_us.query(query)
iterator = query_job.result(page_size=CHUNK_SIZE)
pages = iterator.pages

# Calculer le nombre total de batches à traiter (approximatif basé sur MAX_BATCHES)
total_batches = MAX_BATCHES

# Utiliser tqdm pour la barre de progression sur les pages
for i, page in enumerate(tqdm(pages, total=total_batches, desc="Processing Batches")):
    batch_num = START_FROM_BATCH + i
    rows = list(page)
    if not rows:
        print(f"📦 Batch {batch_num} - empty page, skipping.")
        continue

    df_batch = pd.DataFrame(
        data=[dict(row.items()) for row in rows],
        columns=[field.name for field in iterator.schema]
    )

    # print(f"📦 Batch {batch_num} ({len(df_batch)} lignes) → Destination") # Moins verbeux avec tqdm

    job_config = bigquery.LoadJobConfig(
        # Si la table n'existait pas OU si c'est le PREMIER batch calculé (START_FROM_BATCH)
        # ET que ce premier batch correspond bien au tout début (offset=0)
        write_disposition=(
            "WRITE_TRUNCATE" if not table_exists and batch_num == 1 and offset == 0 else "WRITE_APPEND"
        ),
        autodetect=True,
    )

    job = bq_eu.load_table_from_dataframe(
        df_batch,
        destination=table_ref, # Utiliser table_ref
        job_config=job_config,
    )
    job.result() # Attendre la fin du chargement
    # print(f"✅ Batch {batch_num} inséré") # Moins verbeux avec tqdm
    tqdm.write(f"✅ Batch {batch_num} ({len(df_batch)} lignes) inséré") # Utiliser tqdm.write

print(f"🎉 Importation de {total_batches} batches terminée dans BigQuery Europe.")
