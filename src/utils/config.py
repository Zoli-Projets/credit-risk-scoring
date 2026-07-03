import yaml
from pathlib import Path
from typing import Dict, Any

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Cache pour la configuration
_config_cache = None

def load_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (Path, optional): Path to config file. 
                                      Defaults to CONFIG_PATH.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    global _config_cache
    
    if config_path is None:
        config_path = CONFIG_PATH
    
    if _config_cache is None:
        with open(config_path, 'r', encoding='utf-8') as f:
            _config_cache = yaml.safe_load(f)
    
    return _config_cache

def get_config_section(section: str) -> Dict[str, Any]:
    """
    Get a specific section from the configuration.
    
    Args:
        section (str): Section name (e.g., 'xgboost', 'cv')
    
    Returns:
        Dict[str, Any]: Section configuration
    """
    config = load_config()
    return config.get(section, {})

def get_seed() -> int:
    """Get random seed from configuration."""
    config = load_config()
    return config.get('seed', 42)

def get_test_size() -> float:
    """Get test size from configuration."""
    config = load_config()
    return config.get('test_size', 0.2)

def get_target_column() -> str:
    """Get target column name from configuration."""
    config = load_config()
    return config.get('target', 'default')

# Dossiers
RAW_DATA = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DATA = PROJECT_ROOT / "data" / "external"

MODELS = PROJECT_ROOT / "models"
FIGURES = PROJECT_ROOT / "figures"
NOTEBOOKS = PROJECT_ROOT / "notebooks"
REPORTS = PROJECT_ROOT / "reports"

# Variables globales
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = 'default'

# Pour la compatibilité
__all__ = [
    'PROJECT_ROOT',
    'CONFIG_PATH',
    'load_config',
    'get_config_section',
    'get_seed',
    'get_test_size',
    'get_target_column',
    'RAW_DATA',
    'PROCESSED_DATA',
    'MODELS',
    'FIGURES',
    'RANDOM_STATE',
    'TEST_SIZE',
    'TARGET_COLUMN'
]