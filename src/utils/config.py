from pathlib import Path

# Racine du projet (remonte de 2 niveaux depuis ce fichier)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dossiers principaux
RAW_DATA = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DATA = PROJECT_ROOT / "data" / "external"

# Modèles et figures
MODELS = PROJECT_ROOT / "models"
FIGURES = PROJECT_ROOT / "figures"

# Notebooks et rapports
NOTEBOOKS = PROJECT_ROOT / "notebooks"
REPORTS = PROJECT_ROOT / "reports"

# Variables globales du modèle
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = 'default'   