from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.predict_model import predict_data
from steps.evaluate_model import model_evaluation


@pipeline
def train_pipeline(data_path: str) -> dict:
    df = ingest_data(data_path)
    X_train,X_test,y_train,y_test = clean_data(df)
    model_path, encoder_path = train_model(X_train, y_train)
    y_pred, y_prob = predict_data(X_test, model_path, encoder_path)
    metric = model_evaluation(y_test, y_pred, y_prob)
    return metric

