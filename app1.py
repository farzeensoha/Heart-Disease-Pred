import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

# Helper function to load the model and scaler
def load_model(model_filename, scaler_filename):
    try:
        # Attempt to load the model and scaler using joblib
        log_model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
        return log_model, scaler
    except joblib.externals.loky.process_executor.BrokenProcessPool:
        # If the file is not a valid joblib file, print an error message
        st.error(f"Error: The model file {model_filename} is not a valid joblib file.")
        return None, None
    except FileNotFoundError:
        # If the file is not found, print an error message
        st.error(f"Error: The model file {model_filename} was not found.")
        return None, None
    except Exception as e:
        # Catch any other exceptions
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

# Feature list in correct order
features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

# Medical standards for comparison
medical_standards = {
    "Age": "Varies by individual",
    "RestingBP": "90-120 mmHg",
    "Cholesterol": "<200 mg/dl",
    "MaxHR": "220 - Age",
    "Oldpeak": "0.0 - 1.0 (normal)",
}

# Web scraping function for YouTube video links
def fetch_youtube_links(query):
    search_url = f"https://www.youtube.com/results?search_query={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    video_links = []

    for video in soup.find_all('a', href=True):
        href = video['href']
        if '/watch?v=' in href:
            video_links.append(f"https://www.youtube.com{href}")
        if len(video_links) >= 3:  # Limit to 3 results per query
            break
    return video_links

# Function to make predictions
def make_prediction(input_data, log_model, scaler):
    input_df = pd.DataFrame([input_data], columns=features)
    scaled_input = scaler.transform(input_df)
    prediction = log_model.predict(scaled_input)
    probability = log_model.predict_proba(scaled_input)[0][1]
    return prediction[0], probability

# Generate textual explanations of risk factors
def generate_risk_explanations(input_data):
    explanations = []
    youtube_queries = []
    if input_data[3] > 120:
        explanations.append("🩺 High resting blood pressure indicates increased strain on the heart.")
        youtube_queries.append("how to reduce high blood pressure")
    if input_data[4] > 200:
        explanations.append("🥗 Elevated cholesterol levels can lead to blocked arteries.")
        youtube_queries.append("how to lower cholesterol")
    if input_data[7] < (220 - input_data[0]):
        explanations.append("🏃 Low maximum heart rate may indicate poor cardiovascular fitness.")
        youtube_queries.append("how to improve cardiovascular fitness")
    if input_data[9] > 1.0:
        explanations.append("📉 Elevated ST depression (Oldpeak) suggests ischemic heart changes.")
        youtube_queries.append("understanding ST depression")
    return explanations or ["✅ All input values are within normal ranges."], youtube_queries

# Display a bar chart for the input risk factors
def display_risk_factors_chart(input_data):
    data = {
        "Feature": features,
        "Value": input_data
    }
    df = pd.DataFrame(data)
    fig = px.bar(df, x="Feature", y="Value", title="Input Features and Their Values",
                 labels={"Value": "Input Value"}, text_auto=True, color="Feature")
    st.plotly_chart(fig, use_container_width=True)

# Gauge chart: Likelihood of Heart Disease
def display_probability_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Heart Disease Likelihood (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if probability > 0.5 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "pink"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# Display pie chart for risk factors
def display_risk_factor_pie(input_data):
    deviations = [
        max(0, input_data[3] - 120),  # Resting BP
        max(0, input_data[4] - 200),  # Cholesterol
        max(0, (220 - input_data[0]) - input_data[7]),  # MaxHR deviation
        max(0, input_data[9] - 1.0)  # Oldpeak
    ]
    labels = ["Resting BP", "Cholesterol", "MaxHR", "Oldpeak"]
    
    fig = px.pie(values=deviations, names=labels, title="Risk Factor Contribution")
    st.plotly_chart(fig, use_container_width=True)

# Generate a comparison table
def generate_comparison_table(input_data):
    comparison_data = {
        "Feature": ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"],
        "Input Value": [input_data[0], input_data[3], input_data[4], input_data[7], input_data[9]],
        "Medical Standard": [
            medical_standards["Age"], 
            medical_standards["RestingBP"], 
            medical_standards["Cholesterol"], 
            f"{220 - input_data[0]} bpm", 
            medical_standards["Oldpeak"]
        ]
    }
    return pd.DataFrame(comparison_data)

# Generate a downloadable report
def generate_report(input_data, prediction, probability, risk_explanations, youtube_queries, comparison_table):
    report = (
        "=" * 50 + "\n"
        "🩺 Heart Disease Prediction Report 🩺\n"
        + "=" * 50 + "\n\n"
        "Prediction Result\n"
        + "-" * 50 + "\n"
        + f"* Prediction: {'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'}\n"
        + f"* Likelihood of Heart Disease: {probability * 100:.2f}%\n\n"
        "User Inputs and Comparison to Medical Standards\n"
        + "-" * 50 + "\n"
        + comparison_table.to_string(index=False) + "\n\n"
        "Risk Factor Analysis\n"
        + "-" * 50 + "\n"
        + "\n".join(risk_explanations) + "\n\n"
        "Health Suggestions\n"
        + "-" * 50 + "\n"
        + "\n".join([ 
            "🥗 Eat more fruits and vegetables.", 
            "🏃‍♀ Exercise regularly.", 
            "⚖ Maintain a healthy weight.", 
            "💧 Stay hydrated.", 
            "🚭 Avoid smoking and limit alcohol intake.", 
            "🧘‍♀ Manage stress effectively.", 
            "🩺 Regularly monitor blood pressure and cholesterol.", 
            "😴 Get 7-9 hours of sleep daily.", 
            "⏳ Avoid prolonged sitting; stay active."
        ]) + "\n\n"
        + "=" * 50 + "\n"
    )
    return report

# Streamlit User Interface
def user_interface():
    st.title("❤ Heart Disease Prediction")

    # Collect user inputs
    age = st.number_input("📅 Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("⚤ Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    chest_pain = st.selectbox("💔 Chest Pain Type", options=[0, 1, 2, 3], 
                              format_func=lambda x: ["Typical", "Atypical", "Non-anginal", "Asymptomatic"][x])
    resting_bp = st.number_input("🩺 Resting Blood Pressure (mmHg)", min_value=50, max_value=200, value=120)
    cholesterol = st.number_input("🥗 Cholesterol (mg/dl)", min_value=100, max_value=600, value=180)
    fasting_bs = st.selectbox("🩸 Fasting Blood Sugar (1 = >120 mg/dl, 0 = Normal)", options=[1, 0])
    resting_ecg = st.selectbox("📉 Resting ECG", options=[0, 1, 2], format_func=lambda x: ["Normal", "ST", "LVH"][x])
    max_hr = st.number_input("🏃 Maximum Heart Rate (bpm)", min_value=50, max_value=220, value=170)
    exercise_angina = st.selectbox("🚴 Exercise-Induced Angina (1 = Yes, 0 = No)", options=[1, 0])
    oldpeak = st.number_input("📉 Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=0.0)
    st_slope = st.selectbox("📈 ST Slope", options=[0, 1, 2], 
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])

    input_data = [age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]

    # Load model and scaler
    log_model, scaler = load_model("logreg_model.pkl", "scaler.pkl")
    
    # Prediction triggered only after the user clicks "Predict"
    if st.button("Predict"):
        if log_model and scaler:
            # Make prediction
            prediction, probability = make_prediction(input_data, log_model, scaler)

            # Generate explanations and YouTube links
            risk_explanations, youtube_queries = generate_risk_explanations(input_data)

            # Display visualizations and results
            display_risk_factors_chart(input_data)
            display_probability_gauge(probability)
            display_risk_factor_pie(input_data)

            # Comparison table
            comparison_table = generate_comparison_table(input_data)
            st.dataframe(comparison_table)

            # Generate and display report
            report = generate_report(input_data, prediction, probability, risk_explanations, youtube_queries, comparison_table)
            st.text_area("Heart Disease Prediction Report", report, height=300)

            # Downloadable report
            st.download_button("Download Report", report, file_name="heart_disease_report.txt")

# Run the Streamlit app
if _name_ == "_main_":
    user_interface()
