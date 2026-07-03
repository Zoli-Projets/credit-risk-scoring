import logging

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s - %(levelname)s - %(message)s"

)

logger = logging.getLogger(__name__)

logger.info("Loading data...")

logger.info("Training XGBoost...")

logger.info("Saving best model...")