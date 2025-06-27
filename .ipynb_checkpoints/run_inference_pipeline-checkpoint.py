from pipelines.inference_pipeline import inference_pipeline

file_path_csv = "C:/Users/SPOT/Documents/Customer_Churn/data/telco_churn_with_all_feedback.csv"

if __name__ == "__main__":
    # Run the pipeline
    inference_pipeline(input_data_path=file_path_csv)()
