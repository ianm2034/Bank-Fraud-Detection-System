import streamlit as st
import joblib
import pandas as pd

# Load the saved pipeline
pipeline = joblib.load("fraud_detection_pipeline.pkl")

# Streamlit UI
st.title("Credit Card Fraud Detection")

# Option for Single Transaction or Batch via CSV file
mode = st.radio("Choose input method:", ("Single Transaction", "Batch via CSV"))

# ------------------------------- Single Transaction -------------------------------
if mode == "Single Transaction":
    st.write("Enter the details of the transaction:")

    # Create input fields for user to input transaction data
    amt = st.number_input("Transaction Amount (amt)", min_value=0.0, value=120.0)
    category = st.selectbox("Category", ['food', 'electronics', 'clothing', 'other'])
    gender = st.selectbox("Gender", ['M', 'F'])
    state = st.text_input("State", value="CA")
    city_pop = st.number_input("City Population", min_value=0, value=100000)
    job = st.text_input("Job", value="Engineer")
    lat = st.number_input("Latitude", value=34.0522)
    long = st.number_input("Longitude", value=-118.2437)
    merch_lat = st.number_input("Merchant Latitude", value=34.0522)
    merch_long = st.number_input("Merchant Longitude", value=-118.2437)
    trans_date_trans_time = st.text_input("Transaction Date and Time", value="2023-10-26 12:00:00")
    hour = st.number_input("Hour of Transaction", min_value=0, max_value=23, value=12)
    day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, value=3)
    month = st.number_input("Month", min_value=1, max_value=12, value=10)
    amt_bin = st.selectbox("Amount Bin", ['0-50', '50-200', '200-500', '500-1000', '1000+'])
    distance = st.number_input("Distance", min_value=0.0, value=0.0)

    # Create input dictionary based on user input
    input_data = {
        'amt': amt,
        'category': category,
        'gender': gender,
        'state': state,
        'city_pop': city_pop,
        'job': job,
        'lat': lat,
        'long': long,
        'merch_lat': merch_lat,
        'merch_long': merch_long,
        'trans_date_trans_time': trans_date_trans_time,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'amt_bin': amt_bin,
        'distance': distance
    }

    # Convert to DataFrame for prediction
    new_data = pd.DataFrame([input_data])

    # Predict using the loaded pipeline
    if st.button("Predict Fraudulent Transaction"):
        prediction = pipeline.predict(new_data)
        fraud_status = "Fraudulent" if prediction[0] == 1 else "Legitimate"
        st.write(f"Prediction: {fraud_status}")

        # Optionally, show the probability
        proba = pipeline.predict_proba(new_data)[0][1]
        st.write(f"Fraud probability: {proba:.2%}")

# ------------------------------- Batch Prediction via CSV ------------------------
else:
    st.write("Upload a CSV file containing transaction data:")

    # File upload input
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(uploaded_file)

        # Check that required columns are in the dataset
        required_columns = ['amt', 'category', 'gender', 'state', 'city_pop', 'job', 'lat', 'long', 'merch_lat', 'merch_long', 
                            'trans_date_trans_time', 'hour', 'day_of_week', 'month', 'amt_bin', 'distance']
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"Missing columns: {', '.join(missing_columns)}")
            st.stop()

        # Predict on the uploaded data
        st.write("Making predictions on the uploaded data...")

        predictions = pipeline.predict(data)
        proba = pipeline.predict_proba(data)[:, 1]

        # Add prediction and fraud probability to the dataframe
        data['prediction'] = ["Fraud" if p == 1 else "Legitimate" for p in predictions]
        data['fraud_probability'] = proba

        # Show the result (first 50 rows)
        st.dataframe(data.head(50))

        # Option to download the results
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions as CSV", csv, "fraud_predictions.csv", "text/csv")
