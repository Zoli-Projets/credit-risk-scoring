"""
Module de gestion de la configuration du projet.
Charge et expose les paramètres depuis config/config.yaml
"""

import os
import yaml
from pathlib import Path


class Config:
    """Classe singleton pour la configuration du projet."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Charge la configuration depuis le fichier YAML."""
        # Chercher le fichier config.yaml
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Fichier de configuration non trouvé : {config_path}\n"
                "Assurez-vous que le fichier config/config.yaml existe."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # Ajouter le chemin racine du projet
        self._config["project_root"] = str(project_root)

    def get(self, key, default=None):
        """
        Récupère une valeur de configuration.
        Supporte les clés imbriquées avec le format 'section.subsection.key'.
        """
        if self._config is None:
            return default

        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getattr__(self, name):
        """Permet d'accéder aux paramètres comme des attributs."""
        return self.get(name)

    @property
    def all(self):
        """Retourne toute la configuration."""
        return self._config


# Instance globale de la configuration
config = Config()


# Fonctions de confort pour accéder aux paramètres courants
def get_seed():
    return config.get("seed", 42)


def get_random_state():
    return config.get("random_state", 42)


def get_test_size():
    return config.get("test_size", 0.25)


def get_target():
    return config.get("target", "default")


def get_data_path():
    return config.get("data.path", "data/")


def get_selected_features():
    return config.get("feature_selection.selected_features", [])


def get_cols_to_scale():
    return config.get("scaling.cols_to_scale", [])


def get_xgboost_params():
    return config.get("xgboost", {})


def get_logistic_params():
    return config.get("logistic_regression", {})


def get_optuna_params():
    return config.get("optuna", {})


def get_smote_params():
    return config.get("smote", {})


def get_model_save_path():
    return config.get("model_save.path", "model_data.joblib")