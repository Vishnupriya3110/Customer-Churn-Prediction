# coding: utf-8
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle
import logging

# Initialize Flask app
app = Flask("__name__", template_folder='C:/Users/nagas')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
try:
    model = pickle.load(open("model.sav", "rb"))
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("Model file 'model.sav' not found. Ensure it is in the correct directory.")
    raise

# Load the dataset for feature alignment (if needed)
try:
    df_1 = pd.read_csv("first_telc.csv")
    logging.info("Dataset loaded successfully.")
except FileNotFoundError:
    logging.error("Dataset file 'first_telc.csv' not found. Ensure it is in the correct directory.")
    raise

@app.route("/")
def loadPage():
    # Render the initial home page with no output
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    try:
        # Log incoming form data
        logging.debug(f"Form data received: {request.form}")

        # Collect input data from form
        input_data = [request.form.get(f'query{i + 1}', '') for i in range(19)]
        
        # Validate inputs
        if any(value.strip() == "" for value in input_data):
            return "All fields are required.", 400

        # Create DataFrame from inputs
        data = [input_data]
        new_df = pd.DataFrame(data, columns=[
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
            'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'tenure'
        ])

        # Prepare DataFrame for prediction
        new_df['SeniorCitizen'] = new_df['SeniorCitizen'].astype(int)
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        new_df['tenure_group'] = pd.cut(new_df['tenure'].astype(int), range(1, 80, 12), right=False, labels=labels)
        new_df.drop(columns=['tenure'], inplace=True)

        # One-hot encode categorical variables
        encoded_df = pd.get_dummies(new_df[[
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
        ]])

        # Align columns with training data
        for col in model.feature_names_in_:
            if col not in encoded_df.columns:
                encoded_df[col] = 0

        # Drop extra columns
        encoded_df = encoded_df[model.feature_names_in_]

        # Make predictions
        prediction = model.predict(encoded_df.tail(1))
        probability = model.predict_proba(encoded_df.tail(1))[:, 1][0] * 100

        # Output results
        result = "This customer is likely to churn!" if prediction[0] == 1 else "This customer is likely to stay."
        confidence = f"Confidence: {probability:.2f}%"
        logging.info(f"Prediction result: {result} | {confidence}")

        return render_template('home.html', output1=result, output2=confidence, **request.form.to_dict())

    except Exception as e:
        # Log the error
        logging.error(f"Error during prediction: {e}")
        return f"An error occurred during prediction: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
