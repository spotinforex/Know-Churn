from zenml import pipeline
from steps.load_model import load_model_and_encoder
from steps.run_inference import make_prediction
from steps.load_input_data import load_input

@pipeline
def inference_pipeline(input_data_path: str):
    model, encoder = load_model_and_encoder()
    df = load_input(input_data_path)
    pred_df = make_prediction(df, model, encoder)
    return pred_df