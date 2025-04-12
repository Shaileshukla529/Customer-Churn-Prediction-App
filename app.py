import streamlit as st
import pandas as pd
import joblib # Make sure to import joblib
import numpy as np # Import numpy for potential use in load_data or elsewhere

# --- Define Function to Load Original Data ---
# This function is needed just to get original column names and options for UI
@st.cache_data # Cache the data loading
# CORRECTED FILE PATH HERE: Removed the folder name
def load_data(file_path="Customer_churn_prediction.csv"):
    """Loads and performs minimal cleaning on the original CSV data."""
    try:
        df = pd.read_csv(file_path)
        # Keep customerID temporarily if needed for reference, or drop if not
        # df.drop(columns="customerID", inplace=True, errors='ignore') # errors='ignore' is safer

        # Basic cleaning needed for column checks/options
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Don't dropna here if you need all original options for dropdowns
        # df.dropna(subset=['TotalCharges'], inplace=True)

        # Ensure correct dtypes for columns used in logic/options if needed
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
        # Add other type conversions if necessary for generating options
        return df
    except FileNotFoundError:
        # Updated error message to reflect the corrected path expectation
        st.error(f"Original data file '{file_path}' not found in the same directory as app.py. Please ensure it's there.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading original data: {e}")
        st.stop()

# --- Load the PRE-TRAINED Pipeline and Encoder ---
try:
    # Assuming these .joblib files are also in the same directory as app.py
    pipeline = joblib.load('customer_churn_pipeline_lr.joblib')
    st.success("Pre-trained pipeline loaded successfully!")
except FileNotFoundError:
    st.error("Pipeline file 'customer_churn_pipeline_lr.joblib' not found. Please run the notebook to save it and ensure it's in the same directory as app.py.")
    st.stop() # Stop the app if the model can't be loaded
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()

try:
    label_encoder = joblib.load('label_encoder_lr.joblib')
    st.success("Label encoder loaded successfully!")
except FileNotFoundError:
    st.error("Encoder file 'label_encoder_lr.joblib' not found. Please run the notebook to save it and ensure it's in the same directory as app.py.")
    st.stop() # Stop the app if the encoder can't be loaded
except Exception as e:
    st.error(f"Error loading label encoder: {e}")
    st.stop()


# --- Streamlit App UI ---
st.title("Customer Churn Prediction")

# --- Load Original Data (for UI options/column order ONLY) ---
# Now call load_data AFTER it's defined and AFTER loading the model
df_original = load_data() # Uses the corrected default path now
# Drop customerID now if it exists and isn't needed for prediction features
df_original_features = df_original.drop(columns=['Churn', 'customerID'], errors='ignore')


# Get column names and unique values for dropdowns etc. from original data
try:
    original_cols = df_original_features.columns.tolist() # Get feature columns

    input_options = {}
    for col in original_cols:
         # Check if column exists and is not purely numeric for options
         if col in df_original_features and df_original_features[col].dtype == 'object':
              # Get unique values, handle potential NaN values if not dropped earlier
              input_options[col] = df_original_features[col].dropna().unique().tolist()
         elif col == 'SeniorCitizen': # Handle specific known cases
             input_options[col] = [0, 1] # Use numerical values

except Exception as e:
     st.warning(f"Could not automatically determine all input options: {e}. Using manual defaults.")
     # Define defaults manually if the above fails (ensure these match your data)
     input_options = {
         'gender': ['Male', 'Female'],
         'SeniorCitizen': [0, 1],
         'Partner': ['Yes', 'No'],
         'Dependents': ['Yes', 'No'],
         'PhoneService': ['Yes', 'No'],
         'MultipleLines': ['No phone service', 'No', 'Yes'],
         'InternetService': ['DSL', 'Fiber optic', 'No'],
         'OnlineSecurity': ['No', 'Yes', 'No internet service'],
         'OnlineBackup': ['Yes', 'No', 'No internet service'],
         'DeviceProtection': ['No', 'Yes', 'No internet service'],
         'TechSupport': ['No', 'Yes', 'No internet service'],
         'StreamingTV': ['No', 'Yes', 'No internet service'],
         'StreamingMovies': ['No', 'Yes', 'No internet service'],
         'Contract': ['Month-to-month', 'One year', 'Two year'],
         'PaperlessBilling': ['Yes', 'No'],
         'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
         # Numerical columns don't need options here
     }
     # Reconstruct original_cols based on manual defaults + known numerics
     manual_keys = list(input_options.keys())
     known_numerics = ['tenure', 'MonthlyCharges', 'TotalCharges']
     original_cols = [k for k in manual_keys if k not in known_numerics] + known_numerics


st.header("Enter Customer Details:")

input_data = {}

# Create input fields dynamically or manually
col1, col2 = st.columns(2)

# Use a function to safely get options, falling back to empty list if key missing
def get_opts(key):
    # Ensure the key exists in the dictionary before trying to get options
    return input_options.get(key, []) if key in input_options else []


with col1:
    # Check if options are available before creating the selectbox
    gender_opts = get_opts('gender')
    if gender_opts: input_data['gender'] = st.selectbox("Gender", options=gender_opts)

    sc_opts = get_opts('SeniorCitizen')
    if sc_opts: input_data['SeniorCitizen'] = st.selectbox("Senior Citizen", options=sc_opts, format_func=lambda x: "Yes" if x == 1 else "No")

    partner_opts = get_opts('Partner')
    if partner_opts: input_data['Partner'] = st.selectbox("Partner", options=partner_opts)

    dep_opts = get_opts('Dependents')
    if dep_opts: input_data['Dependents'] = st.selectbox("Dependents", options=dep_opts)

    ps_opts = get_opts('PhoneService')
    if ps_opts: input_data['PhoneService'] = st.selectbox("Phone Service", options=ps_opts)

    ml_opts = get_opts('MultipleLines')
    if ml_opts: input_data['MultipleLines'] = st.selectbox("Multiple Lines", options=ml_opts)

    is_opts = get_opts('InternetService')
    if is_opts: input_data['InternetService'] = st.selectbox("Internet Service", options=is_opts)

    os_opts = get_opts('OnlineSecurity')
    if os_opts: input_data['OnlineSecurity'] = st.selectbox("Online Security", options=os_opts)

    ob_opts = get_opts('OnlineBackup')
    if ob_opts: input_data['OnlineBackup'] = st.selectbox("Online Backup", options=ob_opts)


with col2:
    dp_opts = get_opts('DeviceProtection')
    if dp_opts: input_data['DeviceProtection'] = st.selectbox("Device Protection", options=dp_opts)

    ts_opts = get_opts('TechSupport')
    if ts_opts: input_data['TechSupport'] = st.selectbox("Tech Support", options=ts_opts)

    stv_opts = get_opts('StreamingTV')
    if stv_opts: input_data['StreamingTV'] = st.selectbox("Streaming TV", options=stv_opts)

    smv_opts = get_opts('StreamingMovies')
    if smv_opts: input_data['StreamingMovies'] = st.selectbox("Streaming Movies", options=smv_opts)

    cont_opts = get_opts('Contract')
    if cont_opts: input_data['Contract'] = st.selectbox("Contract", options=cont_opts)

    pb_opts = get_opts('PaperlessBilling')
    if pb_opts: input_data['PaperlessBilling'] = st.selectbox("Paperless Billing", options=pb_opts)

    pm_opts = get_opts('PaymentMethod')
    if pm_opts: input_data['PaymentMethod'] = st.selectbox("Payment Method", options=pm_opts)

    # Numerical Inputs (These don't rely on input_options)
    input_data['tenure'] = st.number_input("Tenure (months)", min_value=0, step=1, value=0)
    input_data['MonthlyCharges'] = st.number_input("Monthly Charges", min_value=0.0, step=0.01, format="%.2f", value=0.0)
    input_data['TotalCharges'] = st.number_input("Total Charges", min_value=0.0, step=0.01, format="%.2f", value=0.0)


# Create DataFrame from inputs
try:
    # Ensure all expected columns are present in input_data before creating DataFrame
    for col in original_cols:
        if col not in input_data:
            # Provide a default value if a widget wasn't created (e.g., due to missing options)
            # You might need more sophisticated default logic depending on the column type
            input_data[col] = 0 if col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'] else (get_opts(col)[0] if get_opts(col) else None)

    input_df = pd.DataFrame([input_data])
    # Ensure column order matches the training data features
    input_df = input_df[original_cols] # Use the list of feature columns derived earlier
except Exception as e:
    st.error(f"Error creating input DataFrame: {e}")
    st.write("Collected Input Data:", input_data) # Show what was collected
    st.write("Expected Columns:", original_cols) # Show what columns were expected
    st.stop()


# Predict Button
if st.button("Predict Churn"):
    try:
        # Add a check for NaN values introduced by missing options, if necessary
        if input_df.isnull().values.any():
             st.warning("Warning: Some input values might be missing. Prediction may be affected.")
             # Optionally, handle NaNs before prediction if your pipeline doesn't
             # input_df = input_df.fillna(some_default_value_or_strategy)

        # Make prediction using the loaded pipeline
        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)

        # Decode prediction using the loaded label encoder
        churn_status = label_encoder.inverse_transform(prediction)[0]

        st.subheader("Prediction Result:")
        if churn_status == 'Yes':
            st.error(f"Predicted Churn: **{churn_status}**")
        else:
            st.success(f"Predicted Churn: **{churn_status}**")

        st.write("Prediction Probability:")
        st.write(f"Probability of No Churn: {prediction_proba[0][0]:.4f}")
        st.write(f"Probability of Yes Churn: {prediction_proba[0][1]:.4f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Input Data used for prediction:")
        st.dataframe(input_df)

# Display original data sample (optional)
# if st.checkbox("Show sample of original data"):
#     st.subheader("Sample of Original Data")
#     st.dataframe(df_original.head())

