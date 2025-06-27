from pipelines.training_pipeline import train_pipeline

file_path_csv = "C:/Users/SPOT/Documents/Customer_Churn/data/telco_churn_with_all_feedback.csv"

if __name__ == "__main__":
    # Run the pipeline
    metric = train_pipeline(data_path = file_path_csv)