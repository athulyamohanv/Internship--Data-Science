import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Load the pre-trained logistic regression model
model = pickle.load(open(r'C:\Users\HP\Desktop\project\logistic_regression_model.pkl', 'rb'))

# Function to encode categorical variables
def encode_input(data):
    le = LabelEncoder()
    data['Policy_Type'] = le.fit_transform(data['Policy_Type'])

    oeAS = OrdinalEncoder(categories=[['Minor', 'Moderate', 'Severe']])
    data['Accident_Severity'] = oeAS.fit_transform(data[['Accident_Severity']])

    oeDR = OrdinalEncoder(categories=[['Clean', 'Minor Offenses', 'Major Offenses']])
    data['Driving_Record'] = oeDR.fit_transform(data[['Driving_Record']])

    return data

# Streamlit app
st.set_page_config(page_title="Attorney Involvement Prediction", layout="wide", page_icon="⚖️")

st.title(" Attorney Involvement in Claims Prediction")
st.markdown("""
Welcome to the **Attorney Involvement Prediction App**. Enter the claim details below and click **Submit** to determine if an attorney is likely to be involved.
""")

st.header("Input Details")
st.markdown("Fill in the following fields:")

# Sidebar inputs
clmsex = st.selectbox('Claimant Gender', [1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female')
clminsur = st.selectbox('Insured Claimant', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
seatbelt = st.selectbox('Seatbelt Used', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
clmage = st.number_input('Claimant Age', min_value=1, max_value=100, value=30)
loss = st.number_input('Financial Loss', min_value=0.0, value=5000.0)
accident_severity = st.selectbox('Accident Severity', ['Minor', 'Moderate', 'Severe'])
claim_approval_status = st.selectbox('Claim Approval Status', [1, 0], format_func=lambda x: 'Approved' if x == 1 else 'Denied')
policy_type = st.selectbox('Policy Type', ['Comprehensive', 'Third-Party'])
driving_record = st.selectbox('Driving Record', ['Clean', 'Minor Offenses', 'Major Offenses'])
claim_amount_requested = st.number_input('Claim Amount Requested', min_value=0.0, value=10000.0)
settlement_amount = st.number_input('Settlement Amount', min_value=0.0, value=2000.0)

# Submit button
if st.button("Submit"):
    # Feature engineering
    settlement_ratio = settlement_amount / claim_amount_requested if claim_amount_requested != 0 else 0
    CLMINSUR_LOSS = clminsur * loss
    seatbelt_accident_severity = seatbelt * (1 if accident_severity == 'Minor' else 2 if accident_severity == 'Moderate' else 3)

    # Prepare input data
    input_data = pd.DataFrame({
        'CLMSEX': [clmsex],
        'CLMAGE': [clmage],
        'LOSS': [loss],
        'Accident_Severity': [accident_severity],
        'Claim_Approval_Status': [claim_approval_status],
        'Policy_Type': [policy_type],
        'Driving_Record': [driving_record],
        'settlement_ratio': [settlement_ratio]
    })

    # Encode input data
    input_data = encode_input(input_data)

    # Prediction
    prediction = model.predict(input_data)

    # Display results in the main page
    st.header("Prediction Result:")
    if prediction[0] == 1:
        st.success(" An attorney is likely to be involved in the claim.")
    else:
        st.info(" An attorney is unlikely to be involved in the claim.")

    # Add a summary of input details
    #st.header("Claim Details Summary:")
    #st.table(input_data)
else:
    st.info("Please fill in the details and click **Submit** to see the prediction.")
