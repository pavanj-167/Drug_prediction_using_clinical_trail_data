import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
@st.cache_resource
def load_model():
    with open('best_model_gb.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Feature names as required by the model
feature_names = [
    'drug_name', 'category', 'duration_days', 'standard_dosage', 'dosage', 'adherence_rate',
    'age', 'gender', 'bmi', 'socioeconomic_score', 'dosage_deviation', 'treatment_compliance_score'
]

# Define mappings for categorical variables
gender_mapping = {'F': 0, 'M': 1}
drug_name_mapping = {
    'Amlodipine': 0, 'Aspirin': 1, 'Gabapentin': 2, 'Levothyroxine': 3, 'Lisinopril': 4,
    'Metformin': 5, 'Montelukast': 6, 'Omeprazole': 7, 'Sertraline': 8, 'Simvastatin': 9
}
category_mapping = {
    'Acid Reflux': 0, 'Asthma': 1, 'Cholesterol': 2, 'Depression': 3, 'Diabetes': 4,
    'Hypertension': 5, 'Neuropathy': 6, 'Pain Management': 7, 'Thyroid': 8
}

# Streamlit interface for the user
st.title("Drug Efficacy Score Prediction")

# Collect user input for new patient data
st.header("Enter New Patient Data:")

# Select drug and category
drug_name = st.selectbox("Drug Name", list(drug_name_mapping.keys()))
drug_category = st.selectbox("Category", list(category_mapping.keys()))

# Input fields for other features
duration_days = st.number_input("Duration Days", min_value=1, max_value=365, value=30)
standard_dosage = st.number_input("Standard Dosage (mg)", min_value=1.0, value=50.0, step=0.1)
dosage = st.number_input("Dosage (mg)", min_value=1.0, value=50.0, step=0.1)
adherence_rate = st.slider("Adherence Rate (%)", min_value=0, max_value=100, value=80)
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", list(gender_mapping.keys()))  # 'M' or 'F'
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
socioeconomic_score = st.number_input("Socioeconomic Score", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

# Create a DataFrame for user input
user_data = pd.DataFrame({
    'drug_name': [drug_name_mapping[drug_name]],  # Encode drug_name
    'category': [category_mapping[drug_category]],  # Encode category
    'duration_days': [duration_days],
    'standard_dosage': [standard_dosage],
    'dosage': [dosage],
    'adherence_rate': [adherence_rate],
    'age': [age],
    'gender': [gender_mapping[gender]],  # Encode gender
    'bmi': [bmi],
    'socioeconomic_score': [socioeconomic_score],
    'dosage_deviation': [0],  # Default value
    'treatment_compliance_score': [0]  # Default value
})

# Load the pre-trained model
model = load_model()

# Add a button for prediction
if st.button("Predict"):
    # Make sure the user data is in the correct order
    user_data = user_data[feature_names]
    
    # Predict the drug efficacy score
    prediction = model.predict(user_data)
    
    # Display the prediction result
    st.write(f"Predicted Drug Efficacy Score: {prediction[0]:.2f}")
