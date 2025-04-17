import os
import yaml

# Chemin par défaut du fichier de configuration
DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../config/pipeline_config.yaml")
)

_config = None

def load_config(config_path: str = DEFAULT_CONFIG_PATH):
    global _config
    if _config is None:
        try:
            with open(config_path, "r") as f:
                _config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement de la configuration: {e}")
    return _config

def get_config():
    """Retourne la configuration chargée (singleton)."""
    if _config is None:
        load_config()
    return _config
