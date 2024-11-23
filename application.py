import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
@st.cache_resource
def load_model():
    with open('best_gradient_boosting_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Feature names as required by the model
feature_names = [
    'drug_name', 'category', 'duration_days', 'standard_dosage', 'dosage', 'adherence_rate',
    'age', 'gender', 'bmi', 'socioeconomic_score', 'dosage_deviation', 'treatment_compliance_score'
]

# Load the dataset to extract unique values for dropdowns
df = pd.read_csv('clinical_trial_dataset.csv')
unique_drug_names = df['drug_name'].unique()
age_min, age_max = int(df['age'].min()), int(df['age'].max())
bmi_min, bmi_max = round(df['bmi'].min(), 1), round(df['bmi'].max(), 1)
socio_min, socio_max = round(df['socioeconomic_score'].min(), 1), round(df['socioeconomic_score'].max(), 1)

# Function to get drug category based on selected drug
def get_drug_category(selected_drug):
    category = df[df['drug_name'] == selected_drug]['category'].iloc[0]
    return category

# Streamlit interface for the user
st.title("Drug Efficacy Score Prediction")

# Collect user input for new patient data
st.header("Enter New Patient Data:")

# Select drug
drug_name = st.selectbox("Drug Name", unique_drug_names)
drug_category = get_drug_category(drug_name)
st.write(f"Category for selected drug: {drug_category}")

# Input fields for other features
duration_days = st.number_input("Duration Days", min_value=1, max_value=365, value=30)
standard_dosage = st.number_input("Standard Dosage (mg)", min_value=1.0, value=50.0, step=0.1)
dosage = st.number_input("Dosage (mg)", min_value=1.0, value=50.0, step=0.1)
adherence_rate = st.slider("Adherence Rate (%)", min_value=0, max_value=100, value=80)
age = st.slider("Age", min_value=age_min, max_value=age_max, value=30)
gender = st.selectbox("Gender", df['gender'].unique())  # 'M' or 'F'
bmi = st.slider("BMI", min_value=bmi_min, max_value=bmi_max, value=25.0)
socioeconomic_score = st.slider("Socioeconomic Score", min_value=socio_min, max_value=socio_max, value=50.0)

# Create a DataFrame for user input
user_data = pd.DataFrame({
    'drug_name': [drug_name],
    'category': [drug_category],
    'duration_days': [duration_days],
    'standard_dosage': [standard_dosage],
    'dosage': [dosage],
    'adherence_rate': [adherence_rate],
    'age': [age],
    'gender': [gender],
    'bmi': [bmi],
    'socioeconomic_score': [socioeconomic_score]
})

# Add the missing columns with default values
user_data['dosage_deviation'] = 0  # Default value, you can adjust as needed
user_data['treatment_compliance_score'] = 0  # Default value, you can adjust as needed

# Label encode the gender column
label_encoder = LabelEncoder()
user_data['gender'] = label_encoder.fit_transform(user_data['gender'])

# Ensure the user input data has the same columns as the model expects
missing_cols = set(feature_names) - set(user_data.columns)
for col in missing_cols:
    if col in df.columns:  # Check if the column exists in the original dataset
        if df[col].dtype == 'object':  # Categorical column
            user_data[col] = df[col].mode()[0]  # Use the most common value
        else:  # Numeric column
            user_data[col] = df[col].mean()  # Use the mean value

# Load the pre-trained model
model = load_model()

# Make sure the user data is in the correct order
user_data = user_data[feature_names]

# Add a button for prediction
if st.button("Predict"):
    # Predict the drug efficacy score
    prediction = model.predict(user_data[feature_names])
    
    # Display the prediction result
    st.write(f"Predicted Drug Efficacy Score: {prediction[0]:.2f}")
