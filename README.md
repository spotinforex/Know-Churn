💼 Know-Churn

An end-to-end churn prediction system built with CatBoost, orchestrated using ZenML, tracked via MLflow, and served through a sleek Streamlit app. Model experimentation was done in Jupyter, and the final model is stored and reused during inference.
 
🧠 Project Highlights
•	🔍 Jupyter notebook for prototyping and model selection
•	📦 ZenML pipelines for training and inference workflows
•	🧮 CatBoostClassifier for final model due to native categorical support
•	📊 MLflow used for experiment tracking and artifact logging
•	🌐 Streamlit frontend for live predictions and user interaction
•	📁 Structured folder system for clarity and scalability
 
🧰 Tech Stack
Purpose	Tool
Model	                       CatBoost
Workflow Orchestration	       ZenML
Experiment Tracking	          MLflow
Notebook Prototyping	       Jupyter
UI	                         Streamlit
Data Handling	           Pandas, Scikit-learn
 
🔧 Getting Started
1. Clone the repository
git clone https://github.com/spotinforex/customer_churn.git
cd customer_churn
2. Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Initialize ZenML and MLflow
zenml init
mlflow ui
Visit http://localhost:5000 to view your experiment logs.
 
📊 Prototyping & Model Selection
Run the notebook:
jupyter notebook jupyter/prototyping.ipynb
The notebook explores:
•	Data distribution
•	Handling missing and categorical values
•	Multiple model comparisons
•	Final choice: CatBoostClassifier
 
🧪 Training the Model
python run_train_pipeline.py
This will:
•	Ingest and clean data
•	Train CatBoost
•	Save model to savedmodel/
•	Log all artifacts to MLflow
 
🔍 Run Inference
python run_inference_pipeline.py
•	Loads saved model and encoder
•	Predicts churn on new input (batch or single)
•	Returns results and logs inference if configured
 
🖥️ Launch Streamlit App
streamlit run streamlit/app.py
What it offers:
•	Upload CSV for bulk predictions
•	Manually enter data for real-time prediction
•	View churn distribution as a bar chart if added to evaluate model
 
📈 MLflow Logging
Each training run logs:
•	Metrics (Accuracy, ROC-AUC, Precision, Recall)
•	Parameters
•	Model artifacts (CatBoost + encoder)
•	Versioning across runs
Logs are stored in the mlruns/ directory and viewable via mlflow ui
 
✅ To-Do / Future Improvements
•	[ ] SHAP explainability visualizations in Streamlit
•	[ ] Switch to cloud storage for model and artifacts
•	[ ] Add user authentication to the app
 
✍️ Author
Praisejah Nwabeke
Data Scientist & Builder & Public Administrator
📧 nwabekepraisejah@gmail.com
 
📜 License
This project is licensed under the MIT License.
