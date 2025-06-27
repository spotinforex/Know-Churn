import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from src.data_cleaning import DataHandler, DataDivideStrategy, DataPreProcessStrategy

# ---------------------- Setup ----------------------
st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("💼 Customer Churn Prediction App")

tab1, tab2 = st.tabs(["📁 Upload CSV", "🧍 Single Customer"])

# ---------------------- Load Model & Encoder ----------------------
@st.cache_resource
def load_model_and_encoder():
    root_dir = Path(__file__).resolve().parents[1]
    model_path = root_dir / "saved_model" / "cat_boost_model.pkl"
    encoder_path = root_dir / "saved_model" / "encoder.pkl"

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

model, encoder = load_model_and_encoder()

# ---------------------- TAB 1: CSV Upload ----------------------
with tab1:
    st.subheader("📁 Batch Prediction via CSV")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔍 Data Preview")
            st.dataframe(raw_df.head())

        with col2:
            if "Churn" in raw_df.columns:
                st.markdown("#### 📊 Churn Distribution")
                fig, ax = plt.subplots()
                sns.countplot(data=raw_df, x="Churn", palette="Set2", ax=ax)
                st.pyplot(fig)
            else:
                st.info("No `Churn` column found to plot distribution.")

        if st.button("🔮 Run Prediction"):
            st.subheader("📋 Prediction Results")
            with st.spinner("Running inference..."):
                try:
                    data_cleaning = DataHandler(data=raw_df, strategy=DataPreProcessStrategy())
                    preprocessed_data = data_cleaning.handle_data()
                    df_encoded = encoder.transform(preprocessed_data)
                    predictions = model.predict(df_encoded)

                    raw_df["Prediction"] = predictions
                    try:
                        raw_df["Probability"] = model.predict_proba(df_encoded)[:, 1]
                    except:
                        pass

                    st.success("✅ Prediction completed!")
                    st.dataframe(raw_df)

                    st.download_button(
                        "⬇️ Download Predictions",
                        raw_df.to_csv(index=False),
                        "predictions.csv"
                    )
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")

# ---------------------- TAB 2: Real-Time Input ----------------------
with tab2:
    st.subheader("🧍 Real-Time Single Customer Prediction")

    input_fields = {
    "gender": st.selectbox("Gender", ["Male", "Female"]),
    "SeniorCitizen": st.selectbox("Senior Citizen", [0, 1]),
    "Partner": st.selectbox("Has Partner?", ["Yes", "No"]),
    "Dependents": st.selectbox("Has Dependents?", ["Yes", "No"]),
    "tenure": st.number_input("Tenure (months)", min_value=0, max_value=100),
    "PhoneService": st.selectbox("Phone Service", ["Yes", "No"]),
    "MultipleLines": st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"]),
    "InternetService": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    "OnlineSecurity": st.selectbox("Online Security", ["Yes", "No", "No internet service"]),
    "OnlineBackup": st.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
    "DeviceProtection": st.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
    "TechSupport": st.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
    "StreamingTV": st.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
    "StreamingMovies": st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
    "Contract": st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": st.selectbox("Paperless Billing", ["Yes", "No"]),
    "PaymentMethod": st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]),
    "MonthlyCharges": st.number_input("Monthly Charges", min_value=0.0),
    "TotalCharges": st.number_input("Total Charges", min_value=0.0),
}


    input_df = pd.DataFrame([input_fields])

    if st.button("🧠 Predict Churn"):
        st.subheader("🔮 Prediction Result")
        try:
            encoded_input = encoder.transform(input_df)
            pred = model.predict(encoded_input)[0]
            prob = model.predict_proba(encoded_input)[0][1]

            st.write(f"**Prediction:** {'Churn' if pred == 1 else 'No Churn'}")
            st.write(f"**Probability of Churn:** {round(prob * 100, 2)}%")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
