# src/pipelines/components/model_training/launch_hpt_job.py

# Correction : Ajout de logs détaillés pour le débogage HPT spec construction

from kfp.dsl import component, Input, Output, Model, Artifact
from typing import Dict, List, NamedTuple, Any
import logging
import os
import traceback # Importer traceback pour afficher les exceptions complètes

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
    worker_pool_spec: Dict,
    enable_hpt: bool,
    default_hyperparameters: Dict[str, Any],
    bq_table_name_path: Input[Artifact],
    static_args: Dict,
    model_gcs_path: Output[Model],
    # Paramètres HPT simples
    hpt_metric_tag: str,
    hpt_metric_goal: str,
    hpt_max_trial_count: int,
    hpt_parallel_trial_count: int,
    hpt_search_algorithm: str,
    hpt_n_estimators_min: int,
    hpt_n_estimators_max: int,
    hpt_n_estimators_scale: str,
    hpt_learning_rate_min: float,
    hpt_learning_rate_max: float,
    hpt_learning_rate_scale: str,
    hpt_max_depth_min: int,
    hpt_max_depth_max: int,
    hpt_max_depth_scale: str,
    hpt_reg_lambda_min: float,
    hpt_reg_lambda_max: float,
    hpt_reg_lambda_scale: str,
    # Ajouter d'autres ici...
) -> NamedTuple("Outputs", [("best_trial_id", str), ("best_rmse", float)]):
    """
    Lance un job Vertex AI HPT ou CustomJob.
    Reconstruit la spécification des paramètres HPT à partir d'arguments simples.
    """
    # Imports internes
    from google.cloud import aiplatform
    # --- MODIFICATION : Importer StudySpec depuis v1.types ---
    from google.cloud.aiplatform_v1.types import StudySpec
    from google.cloud.aiplatform import hyperparameter_tuning as hpt # Gardé pour la classe HyperparameterTuningJob
    import time
    from collections import namedtuple
    import json
    import logging # Importer à nouveau ici
    import traceback # Importer traceback pour afficher les exceptions complètes

    Outputs = namedtuple("Outputs", ["best_trial_id", "best_rmse"])

    # --- DEBUG : Afficher les paramètres HPT simples reçus ---
    logging.info("--- Début launch_hpt_job ---")
    logging.info(f"Paramètres HPT reçus:")
    logging.info(f"  n_estimators: min={hpt_n_estimators_min} (type: {type(hpt_n_estimators_min)}), max={hpt_n_estimators_max} (type: {type(hpt_n_estimators_max)}), scale='{hpt_n_estimators_scale}'")
    logging.info(f"  learning_rate: min={hpt_learning_rate_min} (type: {type(hpt_learning_rate_min)}), max={hpt_learning_rate_max} (type: {type(hpt_learning_rate_max)}), scale='{hpt_learning_rate_scale}'")
    logging.info(f"  max_depth: min={hpt_max_depth_min} (type: {type(hpt_max_depth_min)}), max={hpt_max_depth_max} (type: {type(hpt_max_depth_max)}), scale='{hpt_max_depth_scale}'")
    logging.info(f"  reg_lambda: min={hpt_reg_lambda_min} (type: {type(hpt_reg_lambda_min)}), max={hpt_reg_lambda_max} (type: {type(hpt_reg_lambda_max)}), scale='{hpt_reg_lambda_scale}'")
    logging.info(f"  metric_tag='{hpt_metric_tag}', metric_goal='{hpt_metric_goal}', max_trials={hpt_max_trial_count}, parallel_trials={hpt_parallel_trial_count}, algorithm='{hpt_search_algorithm}'")
    # ---------------------------------------------------------

    aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    display_name = f"{display_name_prefix}-{timestamp}"
    best_trial_id_out = "N/A"
    best_rmse_out = float('nan')
    model_uri = None

    # Lire le nom de la table (logique identique)
    try:
        with open(bq_table_name_path.path, 'r') as f: bq_table_name_str = f.read().strip()
        if not bq_table_name_str: raise ValueError("Fichier nom table vide.")
        logging.info(f"Nom table BQ lu: '{bq_table_name_str}'")
    except Exception as read_e:
        logging.error(f"Erreur lecture nom table: {read_e}")
        raise RuntimeError(f"Impossible lire nom table BQ: {read_e}") from read_e

    # Ajouter nom table aux args statiques (logique identique)
    current_static_args = static_args.copy()
    current_static_args['bq_table'] = bq_table_name_str
    base_args_list = [f"--{key}={value}" for key, value in current_static_args.items()]
    logging.info(f"Args statiques base pour script train: {base_args_list}")

    base_output_dir = f"{staging_bucket}/{display_name}"
    logging.info(f"Répertoire sortie base job: {base_output_dir}")

    if enable_hpt:
        logging.info("Hyperparameter Tuning activé. Configuration du job HPT...")
        script_args = base_args_list + ["--hpt_enabled=True"]
        worker_pool = [
            {"machine_spec": {"machine_type": worker_pool_spec["machine_type"]},
             "replica_count": 1,
             "container_spec": {"image_uri": worker_pool_spec["container_uri"], "args": script_args,},
            }
        ]
        logging.info(f"Worker pool spec: {worker_pool}")

        # --- Reconstruire study_spec_params avec StudySpec.ParameterSpec (types v1) ---
        study_spec_params = []
        logging.info("Reconstruction de HPT parameter spec via types v1 (StudySpec)...")

        # n_estimators (Integer)
        try:
            logging.info("Tentative construction spec 'n_estimators'...")
            scale_enum_value = StudySpec.ParameterSpec.ScaleType[hpt_n_estimators_scale]
            spec = StudySpec.ParameterSpec(
                parameter_id='n_estimators',
                integer_value_spec=StudySpec.ParameterSpec.IntegerValueSpec(
                    min_value=hpt_n_estimators_min, max_value=hpt_n_estimators_max
                ),
                scale_type=scale_enum_value
            )
            study_spec_params.append(spec)
            logging.info(f"  SUCCÈS: Ajouté spec v1 pour n_estimators")
        except KeyError:
            logging.warning(f"  ÉCHELLE INVALIDE (KeyError) '{hpt_n_estimators_scale}' pour n_estimators.")
        except Exception as e:
            logging.error(f"  ERREUR construction spec v1 n_estimators: Type={type(e)}, Msg={e}")
            logging.error(traceback.format_exc())

        # learning_rate (Double)
        try:
            logging.info("Tentative construction spec 'learning_rate'...")
            scale_enum_value = StudySpec.ParameterSpec.ScaleType[hpt_learning_rate_scale]
            spec = StudySpec.ParameterSpec(
                parameter_id='learning_rate',
                double_value_spec=StudySpec.ParameterSpec.DoubleValueSpec(
                    min_value=hpt_learning_rate_min, max_value=hpt_learning_rate_max
                ),
                scale_type=scale_enum_value
            )
            study_spec_params.append(spec)
            logging.info(f"  SUCCÈS: Ajouté spec v1 pour learning_rate")
        except KeyError:
            logging.warning(f"  ÉCHELLE INVALIDE (KeyError) '{hpt_learning_rate_scale}' pour learning_rate.")
        except Exception as e:
            logging.error(f"  ERREUR construction spec v1 learning_rate: Type={type(e)}, Msg={e}")
            logging.error(traceback.format_exc())

        # max_depth (Integer)
        try:
            logging.info("Tentative construction spec 'max_depth'...")
            scale_enum_value = StudySpec.ParameterSpec.ScaleType[hpt_max_depth_scale]
            spec = StudySpec.ParameterSpec(
                parameter_id='max_depth',
                integer_value_spec=StudySpec.ParameterSpec.IntegerValueSpec(
                    min_value=hpt_max_depth_min, max_value=hpt_max_depth_max
                ),
                scale_type=scale_enum_value
            )
            study_spec_params.append(spec)
            logging.info(f"  SUCCÈS: Ajouté spec v1 pour max_depth")
        except KeyError:
            logging.warning(f"  ÉCHELLE INVALIDE (KeyError) '{hpt_max_depth_scale}' pour max_depth.")
        except Exception as e:
            logging.error(f"  ERREUR construction spec v1 max_depth: Type={type(e)}, Msg={e}")
            logging.error(traceback.format_exc())

        # reg_lambda (Double)
        try:
            logging.info("Tentative construction spec 'reg_lambda'...")
            scale_enum_value = StudySpec.ParameterSpec.ScaleType[hpt_reg_lambda_scale]
            spec = StudySpec.ParameterSpec(
                parameter_id='reg_lambda',
                double_value_spec=StudySpec.ParameterSpec.DoubleValueSpec(
                    min_value=hpt_reg_lambda_min, max_value=hpt_reg_lambda_max
                ),
                scale_type=scale_enum_value
            )
            study_spec_params.append(spec)
            logging.info(f"  SUCCÈS: Ajouté spec v1 pour reg_lambda")
        except KeyError:
            logging.warning(f"  ÉCHELLE INVALIDE (KeyError) '{hpt_reg_lambda_scale}' pour reg_lambda.")
        except Exception as e:
            logging.error(f"  ERREUR construction spec v1 reg_lambda: Type={type(e)}, Msg={e}")
            logging.error(traceback.format_exc())

        logging.info(f"Paramètres study spec v1 reconstruits final: {study_spec_params}")

        if not study_spec_params:
             logging.error("La liste study_spec_params est vide après tentatives de construction.")
             raise ValueError("Aucun paramètre HPT valide n'a pu être construit à partir des entrées fournies.")

        # Lancer le job HPT (logique identique)
        hpt_job = aiplatform.HyperparameterTuningJob(
            display_name=display_name,
            custom_job=aiplatform.CustomJob(
                display_name=display_name + "_trial-base",
                worker_pool_specs=worker_pool,
                base_output_directory=base_output_dir, # OK ici car dans HPTJob
            ),
            metric_spec={hpt_metric_tag: hpt_metric_goal},
            parameter_spec=study_spec_params, # Utiliser la liste reconstruite (type v1)
            max_trial_count=hpt_max_trial_count,
            parallel_trial_count=hpt_parallel_trial_count,
            search_algorithm=hpt_search_algorithm,
        )
        logging.info(f"Lancement du job HPT {display_name}...")
        # ... (run, refresh, find best trial, model_uri - reste identique) ...
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
        # ... (Logique CustomJob identique à la version précédente corrigée) ...
        logging.info("Hyperparameter Tuning désactivé. Lancement Custom Job unique...")
        script_args = base_args_list + ["--hpt_enabled=False"]
        for key, value in default_hyperparameters.items(): script_args.append(f"--{key}={value}")
        logging.info(f"Args script pour Custom Job: {script_args}")
        worker_pool = [ # Recréer ici car worker_pool était défini dans le bloc if HPT
            {"machine_spec": {"machine_type": worker_pool_spec["machine_type"]},
             "replica_count": 1,
             "container_spec": {"image_uri": worker_pool_spec["container_uri"], "args": script_args},
            }
        ]
        custom_job = aiplatform.CustomJob(display_name=display_name, worker_pool_specs=worker_pool) # Sans base_output_directory
        logging.info(f"Lancement Custom Job {display_name}...")
        custom_job.run(sync=True)
        logging.info(f"Custom Job {custom_job.resource_name} terminé.")
        custom_job.refresh()
        try:
            if hasattr(custom_job.gca_resource, 'job_spec') and hasattr(custom_job.gca_resource.job_spec, 'base_output_directory') and hasattr(custom_job.gca_resource.job_spec.base_output_directory, 'output_uri_prefix'):
                job_output_dir = custom_job.gca_resource.job_spec.base_output_directory.output_uri_prefix
                logging.info(f"Répertoire sortie récupéré via gca_resource: {job_output_dir}")
                model_uri = f"{job_output_dir}/model/model.joblib"
            else:
                logging.warning("Impossible récupérer base_output_directory via gca_resource. Construction manuelle.")
                model_uri = f"{base_output_dir}/model/model.joblib"
        except AttributeError as ae:
             logging.warning(f"Attribut manquant récupération base_output_directory: {ae}. Construction manuelle.")
             model_uri = f"{base_output_dir}/model/model.joblib"
        logging.info(f"URI modèle construit pour Custom Job: {model_uri}")


    # Écrire l'URI du modèle (logique identique)
    if model_uri is None:
         logging.error("model_uri n'a pas été défini.")
         raise RuntimeError("Échec détermination URI modèle après job entraînement.")
    logging.info(f"URI final meilleur modèle : {model_uri}")
    model_gcs_path.uri = model_uri

    # Retourner infos meilleur essai (logique identique)
    return Outputs(best_trial_id_out, best_rmse_out)