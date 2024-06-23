from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Load the model
model = joblib.load('xgboost_model.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data)
        
        # Convert numerical columns to the correct data type
        df['step'] = df['step'].astype(int)
        df['amount'] = df['amount'].astype(float)
        df['oldbalanceOrg'] = df['oldbalanceOrg'].astype(float)
        df['newbalanceOrig'] = df['newbalanceOrig'].astype(float)
        df['oldbalanceDest'] = df['oldbalanceDest'].astype(float)
        df['newbalanceDest'] = df['newbalanceDest'].astype(float)
        
        # One-hot encode 'type' column and ensure all expected columns are present
        df = pd.get_dummies(df, columns=['type'])
        expected_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match the training set
        df = df[expected_columns]
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(df)
        
        # Predict
        prediction = model.predict(dmatrix)
        result = [1 if y > 0.5 else 0 for y in prediction]
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
