# src/pipelines/components/model_training/launch_hpt_job.py

# Correction : Recevoir les specs HPT comme paramètres simples et reconstruire

from kfp.dsl import component, Input, Output, Model, Artifact
# --- MODIFICATION : Any ajouté ---
from typing import Dict, List, NamedTuple, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@component(
    base_image="europe-west1-docker.pkg.dev/avisia-certification-ml-yde/chicago-taxis-demo/forecasting-pipeline:latest",
    # packages_to_install=["google-cloud-aiplatform"]
)
def launch_hpt_job(
    project: str,
    location: str,
    staging_bucket: str,
    display_name_prefix: str,
    worker_pool_spec: Dict, # OK
    enable_hpt: bool, # OK
    # --- MODIFICATION : Supprimer hpt_config ---
    # hpt_config: Dict, # SUPPRIMÉ
    default_hyperparameters: Dict[str, Any], # OK
    bq_table_name_path: Input[Artifact], # OK
    static_args: Dict, # OK (sans bq_table initialement)
    model_gcs_path: Output[Model], # OK

    # --- MODIFICATION : Ajouter les paramètres simples pour HPT ---
    hpt_metric_tag: str,
    hpt_metric_goal: str,
    hpt_max_trial_count: int,
    hpt_parallel_trial_count: int,
    hpt_search_algorithm: str,
    # Paramètres pour n_estimators
    hpt_n_estimators_min: int,
    hpt_n_estimators_max: int,
    hpt_n_estimators_scale: str,
    # Paramètres pour learning_rate
    hpt_learning_rate_min: float,
    hpt_learning_rate_max: float,
    hpt_learning_rate_scale: str,
    # Paramètres pour max_depth
    hpt_max_depth_min: int,
    hpt_max_depth_max: int,
    hpt_max_depth_scale: str,
    # Paramètres pour reg_lambda
    hpt_reg_lambda_min: float,
    hpt_reg_lambda_max: float,
    hpt_reg_lambda_scale: str,
    # Ajouter d'autres ici...

) -> NamedTuple("Outputs", [("best_trial_id", str), ("best_rmse", float)]):
    """
    Lance un job Vertex AI HPT ou CustomJob.
    Reconstruit la spécification des paramètres HPT à partir d'arguments simples.
    [...]
    Args:
        [...]
        # hpt_config: SUPPRIMÉ
        default_hyperparameters: Dictionnaire des HPs par défaut (si HPT désactivé).
        bq_table_name_path: Artefact d'entrée contenant le nom de la table BQ.
        static_args: Arguments statiques pour le script d'entraînement (excluant bq_table).
        model_gcs_path: Artefact de sortie pour l'URI du modèle.
        hpt_metric_tag: Tag de la métrique HPT.
        hpt_metric_goal: Objectif HPT (MINIMIZE/MAXIMIZE).
        hpt_max_trial_count: Nombre max d'essais.
        hpt_parallel_trial_count: Nombre d'essais parallèles.
        hpt_search_algorithm: Algorithme de recherche HPT.
        hpt_n_estimators_min: Borne min pour n_estimators.
        hpt_n_estimators_max: Borne max pour n_estimators.
        hpt_n_estimators_scale: Échelle pour n_estimators.
        # ... (autres paramètres HPT simples) ...
    Returns:
        NamedTuple contenant best_trial_id et best_rmse.
    """
    # Imports internes
    from google.cloud import aiplatform
    from google.cloud.aiplatform import hyperparameter_tuning as hpt
    import time
    from collections import namedtuple
    import json # Garder json pour analyse potentielle d'args si nécessaire
    import logging

    Outputs = namedtuple("Outputs", ["best_trial_id", "best_rmse"])

    aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    display_name = f"{display_name_prefix}-{timestamp}"
    best_trial_id_out = "N/A"
    best_rmse_out = float('nan')
    model_uri = None

    # Lire le nom de la table depuis le fichier d'artefact (logique identique)
    try:
        with open(bq_table_name_path.path, 'r') as f:
            bq_table_name_str = f.read().strip()
        if not bq_table_name_str: raise ValueError("Fichier nom table vide.")
        logging.info(f"Nom table BQ lu: '{bq_table_name_str}'")
    except Exception as read_e:
        logging.error(f"Erreur lecture nom table depuis {bq_table_name_path.path}: {read_e}")
        raise RuntimeError(f"Impossible lire nom table BQ: {read_e}") from read_e

    # Ajouter le nom lu aux arguments statiques (logique identique)
    current_static_args = static_args.copy()
    current_static_args['bq_table'] = bq_table_name_str
    base_args_list = [f"--{key}={value}" for key, value in current_static_args.items()]
    logging.info(f"Args statiques base pour script train: {base_args_list}")

    # Répertoire de sortie (logique identique)
    base_output_dir = f"{staging_bucket}/{display_name}"
    logging.info(f"Répertoire sortie base job: {base_output_dir}")

    if enable_hpt:
        logging.info("Hyperparameter Tuning activé. Configuration du job HPT...")
        script_args = base_args_list + ["--hpt_enabled=True"]

        # Préparer worker_pool (logique identique)
        worker_pool = [
            {"machine_spec": {"machine_type": worker_pool_spec["machine_type"]},
             "replica_count": 1,
             "container_spec": {"image_uri": worker_pool_spec["container_uri"], "args": script_args,},
            }
        ]
        logging.info(f"Worker pool spec: {worker_pool}")

        # --- MODIFICATION : Reconstruire study_spec_params à partir des args simples ---
        study_spec_params = []
        logging.info("Reconstruction de HPT parameter spec à partir des arguments simples...")

        # n_estimators (Integer)
        try: # Utiliser try/except pour robustesse des conversions d'échelle
            scale_n_est = getattr(hpt.ScaleType, hpt_n_estimators_scale, hpt.ScaleType.UNIT_LINEAR_SCALE)
            study_spec_params.append(hpt.IntegerParameterSpec(
                parameter_id='n_estimators', min_value=hpt_n_estimators_min, max_value=hpt_n_estimators_max, scale_type=scale_n_est
            ))
            logging.info(f"Ajouté spec pour n_estimators ({hpt_n_estimators_min}-{hpt_n_estimators_max}, scale={hpt_n_estimators_scale})")
        except AttributeError: logging.warning(f"Échelle invalide '{hpt_n_estimators_scale}' pour n_estimators. Utilisation de UNIT_LINEAR_SCALE.")
        except Exception as e: logging.error(f"Erreur construction spec n_estimators: {e}")

        # learning_rate (Double)
        try:
            scale_lr = getattr(hpt.ScaleType, hpt_learning_rate_scale, hpt.ScaleType.UNIT_LINEAR_SCALE)
            study_spec_params.append(hpt.DoubleParameterSpec(
                parameter_id='learning_rate', min_value=hpt_learning_rate_min, max_value=hpt_learning_rate_max, scale_type=scale_lr
            ))
            logging.info(f"Ajouté spec pour learning_rate ({hpt_learning_rate_min}-{hpt_learning_rate_max}, scale={hpt_learning_rate_scale})")
        except AttributeError: logging.warning(f"Échelle invalide '{hpt_learning_rate_scale}' pour learning_rate. Utilisation de UNIT_LINEAR_SCALE.")
        except Exception as e: logging.error(f"Erreur construction spec learning_rate: {e}")

        # max_depth (Integer)
        try:
            scale_depth = getattr(hpt.ScaleType, hpt_max_depth_scale, hpt.ScaleType.UNIT_LINEAR_SCALE)
            study_spec_params.append(hpt.IntegerParameterSpec(
                parameter_id='max_depth', min_value=hpt_max_depth_min, max_value=hpt_max_depth_max, scale_type=scale_depth
            ))
            logging.info(f"Ajouté spec pour max_depth ({hpt_max_depth_min}-{hpt_max_depth_max}, scale={hpt_max_depth_scale})")
        except AttributeError: logging.warning(f"Échelle invalide '{hpt_max_depth_scale}' pour max_depth. Utilisation de UNIT_LINEAR_SCALE.")
        except Exception as e: logging.error(f"Erreur construction spec max_depth: {e}")

        # reg_lambda (Double)
        try:
            scale_lambda = getattr(hpt.ScaleType, hpt_reg_lambda_scale, hpt.ScaleType.UNIT_LINEAR_SCALE)
            study_spec_params.append(hpt.DoubleParameterSpec(
                parameter_id='reg_lambda', min_value=hpt_reg_lambda_min, max_value=hpt_reg_lambda_max, scale_type=scale_lambda
            ))
            logging.info(f"Ajouté spec pour reg_lambda ({hpt_reg_lambda_min}-{hpt_reg_lambda_max}, scale={hpt_reg_lambda_scale})")
        except AttributeError: logging.warning(f"Échelle invalide '{hpt_reg_lambda_scale}' pour reg_lambda. Utilisation de UNIT_LINEAR_SCALE.")
        except Exception as e: logging.error(f"Erreur construction spec reg_lambda: {e}")

        # Ajouter la logique pour d'autres hyperparamètres ici si nécessaire...

        logging.info(f"Paramètres study spec reconstruits : {study_spec_params}")
        # --- Fin de la reconstruction ---

        # Vérifier si study_spec_params est vide (si toutes les constructions ont échoué)
        if not study_spec_params:
             raise ValueError("Aucun paramètre HPT valide n'a pu être construit à partir des entrées fournies.")

        # Lancer le job HPT en utilisant les specs reconstruites
        hpt_job = aiplatform.HyperparameterTuningJob(
            display_name=display_name,
            custom_job=aiplatform.CustomJob(
                display_name=display_name + "_trial-base",
                worker_pool_specs=worker_pool,
                base_output_directory=base_output_dir,
            ),
            # Utiliser les paramètres simples directement pour la config générale
            metric_spec={hpt_metric_tag: hpt_metric_goal},
            parameter_spec=study_spec_params, # Utiliser la liste reconstruite
            max_trial_count=hpt_max_trial_count,
            parallel_trial_count=hpt_parallel_trial_count,
            search_algorithm=hpt_search_algorithm,
        )

        logging.info(f"Lancement du job HPT {display_name}...")
        # ... (Reste du code HPT : run, refresh, find best trial, model_uri) ...
        hpt_job.run(sync=True)
        logging.info(f"Job HPT {hpt_job.resource_name} terminé.")
        hpt_job.refresh()

        valid_trials = [t for t in hpt_job.trials if t.state == aiplatform.gapic.Trial.State.SUCCEEDED and t.final_measurement]
        if not valid_trials: raise RuntimeError("Aucun essai HPT réussi.")
        logging.info(f"Trouvé {len(valid_trials)} essais réussis.")
        try:
            if hpt_metric_goal == "MAXIMIZE": best_trial = max(valid_trials, key=lambda t: t.final_measurement.metrics[0].value)
            else: best_trial = min(valid_trials, key=lambda t: t.final_measurement.metrics[0].value)
        except (IndexError, KeyError): raise RuntimeError("Échec détermination meilleur essai (métrique manquante).")

        best_trial_id_out = best_trial.id
        best_rmse_out = best_trial.final_measurement.metrics[0].value
        logging.info(f"Meilleur essai: ID={best_trial_id_out}, Métrique ({hpt_metric_tag}): {best_rmse_out}")
        model_uri = f"{base_output_dir}/{best_trial.id}/model/model.joblib"
        logging.info(f"URI meilleur modèle: {model_uri}")


    else: # Cas Custom Job simple (non HPT)
        logging.info("Hyperparameter Tuning désactivé. Lancement Custom Job unique...")
        # Utiliser base_args_list mis à jour
        script_args = base_args_list + ["--hpt_enabled=False"]
        # Ajouter les HPs par défaut (ceux définis dans le pipeline et passés via default_hyperparameters)
        for key, value in default_hyperparameters.items():
            script_args.append(f"--{key}={value}")
        logging.info(f"Args script pour Custom Job: {script_args}")

        # Préparer worker_pool (logique identique)
        worker_pool = [
            {"machine_spec": {"machine_type": worker_pool_spec["machine_type"]},
             "replica_count": 1,
             "container_spec": {"image_uri": worker_pool_spec["container_uri"], "args": script_args},
            }
        ]
        # Définir et lancer Custom Job (logique identique)
        custom_job = aiplatform.CustomJob(display_name=display_name, worker_pool_specs=worker_pool)
        logging.info(f"Lancement Custom Job {display_name}...")
        custom_job.run(sync=True)
        logging.info(f"Custom Job {custom_job.resource_name} terminé.")
        custom_job.refresh()
        job_output_dir = custom_job.gca_resource.job_spec.base_output_directory.output_uri_prefix
        model_uri = f"{job_output_dir}/model/model.joblib"
        logging.info(f"URI modèle pour Custom Job: {model_uri}")


    # Écrire l'URI du modèle (logique identique)
    if model_uri is None:
         raise RuntimeError("Échec détermination URI modèle après job entraînement.")
    logging.info(f"URI final meilleur modèle : {model_uri}")
    model_gcs_path.uri = model_uri

    # Retourner infos meilleur essai (logique identique)
    return Outputs(best_trial_id_out, best_rmse_out)