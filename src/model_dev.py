import logging 
from abc import ABC, abstractmethod 
from catboost import CatBoostClassifier 
from src.encoder import Encoder
import joblib
import os 
import pandas as pd
import mlflow
import mlflow.catboost
from typing import Tuple
from pathlib import Path




class Model(ABC):
    '''Abstract class for all models'''
    @abstractmethod 
    def train(self, X_train, y_train):
        pass


class CatBoost(Model):
    '''
    CatBoost model with MLflow tracking
    '''
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Tuple[str, str]:
        try:
            logging.info("Model Training in Progress")

            # ========== Preprocess ==========
            encoder = Encoder()
            X_train_encoded = encoder.fit_transform(X_train)

            # ========== Start MLflow run ==========
            mlflow.set_experiment("Churn_Model")
            with mlflow.start_run(run_name="CatBoost_Churn_Model", nested = True):
                logging.info(f"Training with params: {kwargs}")
                mlflow.log_params(kwargs)

                # ========== Train ==========
                model = CatBoostClassifier(**kwargs)
                model.fit(X_train_encoded, y_train)
                logging.info("Model training completed")

                # ========== Log Metric ==========
                acc = model.score(X_train_encoded, y_train)
                mlflow.log_metric("train_accuracy", acc)

                # ========== Save Locally ==========
                save_dir = Path(__file__).resolve().parents[1] / "saved_model"
                save_dir.mkdir(parents=True, exist_ok=True)

                model_path = os.path.join(save_dir, "cat_boost_model.pkl")
                encoder_path = os.path.join(save_dir, "encoder.pkl")

                joblib.dump(model, model_path)
                joblib.dump(encoder, encoder_path)
                logging.info("Model and encoder saved successfully")

                # ========== Log Artifacts to MLflow ==========
                mlflow.catboost.log_model(model, artifact_path="model")
                mlflow.log_artifact(encoder_path, artifact_path="encoder")

                
                return model_path, encoder_path

        except Exception as e:
            logging.exception("Exception during model training")
            raise e
