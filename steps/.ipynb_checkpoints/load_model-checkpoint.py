from zenml import step
import joblib
from pathlib import Path
import logging
from typing import Tuple
from catboost import CatBoostClassifier
from src.encoder import Encoder 

@step
def load_model_and_encoder() -> Tuple[CatBoostClassifier, Encoder]:
    '''
    Loads the trained CatBoost model and encoder for inference.

    Returns:
        Tuple[CatBoostClassifier, Encoder]: Loaded model and encoder instances.
    '''
    try:
        root_dir = Path(__file__).resolve().parents[1]
        model_path = (root_dir / "saved_model" / "cat_boost_model.pkl").resolve()
        encoder_path = (root_dir / "saved_model" / "encoder.pkl").resolve()

        logging.info(f"Loading model from: {model_path}")
        logging.info(f"Loading encoder from: {encoder_path}")

        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)

        logging.info("Model and encoder loaded successfully.")
        return model, encoder

    except Exception as e:
        logging.exception(f"An error occurred while loading model and encoder: {e}")
        raise e
