from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

# Initialize the Flask App
app = Flask(__name__)

# Load the saved model and column names
model = pickle.load(open('fraud_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

@app.route('/')
def home():
    # Renders the index.html page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Retrieve values from the form
    # We create a dictionary to match our model's expectations
    features = {
        'months_as_customer': int(request.form['months_as_customer']),
        'age': int(request.form['age']),
        'policy_deductable': int(request.form['policy_deductable']),
        'total_claim_amount': int(request.form['total_claim_amount'])
    }
    
    # 2. Prepare the data for the model (Handling One-Hot Encoding)
    # We create a dataframe with zeros for all columns
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0
    
    # Fill in the numerical values
    for key, value in features.items():
        if key in model_columns:
            input_df.at[0, key] = value
            
    # Handle the Categorical value (Incident Severity)
    severity = request.form['incident_severity']
    severity_col = f"incident_severity_{severity}"
    if severity_col in model_columns:
        input_df.at[0, severity_col] = 1

    # 3. Make Prediction
    prediction = model.predict(input_df)[0]
    
    # Determine the output text
    if prediction == 1:
        result = "🚩 Warning: This claim is likely FRAUDULENT."
    else:
        result = "✅ Safe: This claim appears to be GENUINE."

    # 4. Render the result back to the same page
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)