import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to handle missing values
def handling_missing_values(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Impute numeric columns with median
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    
    # Fill categorical columns with mode
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    return df

# Function to remove outliers using IQR
def remove_outliers(df, numeric_cols):
    for column in numeric_cols:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Function to calculate VIF
def calculate_vif(df):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    vif_data = pd.DataFrame()
    vif_data['feature'] = df.columns
    vif_data['VIF'] = [variance_inflation_factor(data_scaled, i) for i in range(data_scaled.shape[1])]
    return vif_data

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    columns_to_drop = ['treatment_id', 'patient_id', 'drug_id', 'insurance_type', 'region']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return df

# Training function
def train_model(df):
    # Handle missing values
    df = handling_missing_values(df)

    # Remove outliers
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df = remove_outliers(df, numeric_cols)

    # Feature engineering
    df['dosage_deviation'] = (df['dosage'] - df['standard_dosage']) / df['standard_dosage']
    df['treatment_compliance_score'] = df['adherence_rate'] * (df['duration_days'] / df['duration_days'].max())
    
    # Encoding categorical variables
    label_encoder = LabelEncoder()
    for col in ['gender', 'drug_name', 'category']:
        if col in df.columns:
            df[col] = label_encoder.fit_transform(df[col])

    # Split features and target
    X = df.drop(columns=['efficacy_score'])
    y = df['efficacy_score']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    # Define model
    gb_model = GradientBoostingRegressor(random_state=43)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 4],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', None]
    }
    grid_search_gb = GridSearchCV(
        estimator=gb_model, 
        param_grid=param_grid, 
        cv=3, 
        scoring='neg_mean_squared_error',
        verbose=1, 
        n_jobs=-1
    )

    # Train and find the best model
    grid_search_gb.fit(X_train, y_train)
    best_model = grid_search_gb.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Parameters: {grid_search_gb.best_params_}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R²): {r2}")

    # Save the model
    with open('best_gradient_boosting_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    print("Model training complete and saved as 'best_gradient_boosting_model.pkl'.")

if __name__ == "__main__":
    file_path = r"C:\Users\jammalapavan\Desktop\Streamlit\clinical_trial_dataset.csv"
    df = load_data(file_path)
    train_model(df)
