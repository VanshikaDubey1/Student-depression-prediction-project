from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

app = Flask(__name__)

# Global variables to store model and encoders
model = None
label_encoders = {}
feature_columns = []

def load_and_train_model():
    """Load data, train model, and save encoders"""
    global model, label_encoders, feature_columns
    
    try:
        # Load the dataset
        df = pd.read_csv('student_depression_dataset.csv')
        
        # Store original categorical columns for encoding
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'Depression':
                categorical_cols.append(col)
        
        # Initialize and fit label encoders
        df_encoded = df.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Prepare features and target
        X = df_encoded.drop('Depression', axis=1)
        y = df_encoded['Depression']
        feature_columns = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully with accuracy: {accuracy:.4f}")
        
        # Save model and encoders
        joblib.dump(model, 'depression_model.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        joblib.dump(feature_columns, 'feature_columns.pkl')
        
        return True
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

def load_saved_model():
    """Load saved model and encoders"""
    global model, label_encoders, feature_columns
    
    try:
        if os.path.exists('depression_model.pkl'):
            model = joblib.load('depression_model.pkl')
            label_encoders = joblib.load('label_encoders.pkl')
            feature_columns = joblib.load('feature_columns.pkl')
            print("Saved model loaded successfully")
            return True
    except Exception as e:
        print(f"Error loading saved model: {str(e)}")
    
    return False

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Depression Prediction</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #4a5568;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #2d3748;
        }
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 18px;
            width: 100%;
            margin-top: 20px;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        #result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
        }
        .depression-yes {
            background-color: #fed7d7;
            color: #c53030;
            border: 2px solid #fc8181;
        }
        .depression-no {
            background-color: #c6f6d5;
            color: #2f855a;
            border: 2px solid #68d391;
        }
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
        }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 600px) {
            .form-row {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Student Depression Prediction</h1>
        <form id="predictionForm">
            <div class="form-row">
                <div class="form-group">
                    <label for="gender">Gender:</label>
                    <select id="gender" name="gender" required>
                        <option value="">Select Gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="age">Age:</label>
                    <input type="number" id="age" name="age" min="18" max="60" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="city">City:</label>
                    <input type="text" id="city" name="city" required>
                </div>
                <div class="form-group">
                    <label for="profession">Profession:</label>
                    <select id="profession" name="profession" required>
                        <option value="">Select Profession</option>
                        <option value="Student">Student</option>
                        <option value="Working Professional">Working Professional</option>
                    </select>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="academic_pressure">Academic Pressure (1-5):</label>
                    <input type="number" id="academic_pressure" name="academic_pressure" min="1" max="5" step="0.1" required>
                </div>
                <div class="form-group">
                    <label for="work_pressure">Work Pressure (0-5):</label>
                    <input type="number" id="work_pressure" name="work_pressure" min="0" max="5" step="0.1" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="cgpa">CGPA:</label>
                    <input type="number" id="cgpa" name="cgpa" min="0" max="10" step="0.01" required>
                </div>
                <div class="form-group">
                    <label for="study_satisfaction">Study Satisfaction (1-5):</label>
                    <input type="number" id="study_satisfaction" name="study_satisfaction" min="1" max="5" step="0.1" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="job_satisfaction">Job Satisfaction (0-5):</label>
                    <input type="number" id="job_satisfaction" name="job_satisfaction" min="0" max="5" step="0.1" required>
                </div>
                <div class="form-group">
                    <label for="sleep_duration">Sleep Duration:</label>
                    <select id="sleep_duration" name="sleep_duration" required>
                        <option value="">Select Sleep Duration</option>
                        <option value="Less than 5 hours">Less than 5 hours</option>
                        <option value="5-6 hours">5-6 hours</option>
                        <option value="7-8 hours">7-8 hours</option>
                        <option value="More than 8 hours">More than 8 hours</option>
                    </select>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="dietary_habits">Dietary Habits:</label>
                    <select id="dietary_habits" name="dietary_habits" required>
                        <option value="">Select Dietary Habits</option>
                        <option value="Healthy">Healthy</option>
                        <option value="Moderate">Moderate</option>
                        <option value="Unhealthy">Unhealthy</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="degree">Degree:</label>
                    <input type="text" id="degree" name="degree" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="suicidal_thoughts">Have you ever had suicidal thoughts?</label>
                    <select id="suicidal_thoughts" name="suicidal_thoughts" required>
                        <option value="">Select</option>
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="work_study_hours">Work/Study Hours:</label>
                    <input type="number" id="work_study_hours" name="work_study_hours" min="0" max="24" step="0.1" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="financial_stress">Financial Stress:</label>
                    <select id="financial_stress" name="financial_stress" required>
                        <option value="">Select Financial Stress Level</option>
                        <option value="0">0 - No Stress</option>
                        <option value="1">1 - Low Stress</option>
                        <option value="2">2 - Moderate Stress</option>
                        <option value="3">3 - High Stress</option>
                        <option value="4">4 - Very High Stress</option>
                        <option value="5">5 - Extreme Stress</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="family_history">Family History of Mental Illness:</label>
                    <select id="family_history" name="family_history" required>
                        <option value="">Select</option>
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                    </select>
                </div>
            </div>
            
            <button type="submit">Predict Depression Risk</button>
        </form>
        
        <div class="loading" id="loading">
            <p>🔄 Analyzing...</p>
        </div>
        
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData.entries());
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').innerHTML = '';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                if (result.success) {
                    const resultDiv = document.getElementById('result');
                    const prediction = result.prediction;
                    const probability = result.probability;
                    
                    if (prediction === 1) {
                        resultDiv.innerHTML = `
                            <div class="depression-yes">
                                ⚠️ High Risk of Depression Detected<br>
                                Confidence: ${(probability * 100).toFixed(1)}%<br>
                                <small>Please consider consulting a mental health professional.</small>
                            </div>
                        `;
                        resultDiv.className = 'depression-yes';
                    } else {
                        resultDiv.innerHTML = `
                            <div class="depression-no">
                                ✅ Low Risk of Depression<br>
                                Confidence: ${((1 - probability) * 100).toFixed(1)}%<br>
                                <small>Continue maintaining healthy habits!</small>
                            </div>
                        `;
                        resultDiv.className = 'depression-no';
                    }
                } else {
                    document.getElementById('result').innerHTML = `
                        <div style="background-color: #fed7d7; color: #c53030; padding: 15px; border-radius: 8px;">
                            Error: ${result.error}
                        </div>
                    `;
                }
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').innerHTML = `
                    <div style="background-color: #fed7d7; color: #c53030; padding: 15px; border-radius: 8px;">
                        Error: Unable to make prediction. Please try again.
                    </div>
                `;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input data"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        # Create DataFrame with the input data
        input_data = pd.DataFrame([data])
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in input_data.columns:
                try:
                    input_data[col] = encoder.transform(input_data[col])
                except ValueError:
                    # Handle unseen categories by using the most frequent category
                    input_data[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Ensure all required columns are present and in correct order
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)
        
        # Convert Financial Stress to numeric if it's a string
        if 'Financial Stress' in input_data.columns:
            input_data['Financial Stress'] = pd.to_numeric(input_data['Financial Stress'], errors='coerce')
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Get probability for positive class (depression = 1)
        depression_probability = probability[1] if len(probability) > 1 else probability[0]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': float(depression_probability),
            'message': 'High risk of depression' if prediction == 1 else 'Low risk of depression'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model-info')
def model_info():
    """Get information about the model"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'})
        
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            'model_type': 'Random Forest Classifier',
            'n_features': len(feature_columns),
            'feature_columns': feature_columns,
            'top_features': sorted_features[:10],
            'encoders': list(label_encoders.keys())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Starting Student Depression Prediction App...")
    
    # Try to load saved model first, otherwise train new model
    if not load_saved_model():
        print("No saved model found. Training new model...")
        if not load_and_train_model():
            print("Failed to train model. Please check if 'student_depression_dataset.csv' exists.")
            exit(1)
    
    print("Model loaded successfully!")
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)