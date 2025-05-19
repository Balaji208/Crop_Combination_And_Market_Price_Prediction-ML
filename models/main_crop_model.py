import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import warnings

# Suppress sklearn warnings (optional, remove if you want to see them)
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Load and Train the Model (Run Once)
def train_and_save_model():
    # Load the dataset
    data = pd.read_csv('../datasets/Crop_recommendation.csv')  # Your file path
    
    # Features and target
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    # Save the model
    with open('main_crop_model.pkl', 'wb') as file:
        pickle.dump(rf_model, file)
    print("Model saved as 'main_crop_model.pkl'")
    
    return rf_model

# Step 2: Define Realistic Ranges
realistic_ranges = {
    'N': (0, 200), 'P': (0, 200), 'K': (0, 250), 'temperature': (5, 50),
    'humidity': (0, 100), 'ph': (3, 11), 'rainfall': (0, 500)
}

# Step 3: Input Validation and Preprocessing
def validate_and_preprocess_input(N, P, K, temperature, humidity, ph, rainfall):
    inputs = {'N': N, 'P': P, 'K': K, 'temperature': temperature, 
              'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
    
    # Check and convert input types
    for param, val in inputs.items():
        try:
            inputs[param] = float(val)  # Convert to float, catch invalid types
        except (ValueError, TypeError):
            return False, f"Invalid input: {param} must be a number"
    
    # Validate and cap values
    capped_inputs = {}
    warnings_list = []
    for param, val in inputs.items():
        min_val, max_val = realistic_ranges[param]
        if val < min_val or val > max_val:
            warnings_list.append(f"{param} ({val}) outside realistic range ({min_val}-{max_val}), capped")
            capped_inputs[param] = max(min_val, min(val, max_val))
        else:
            capped_inputs[param] = val
    
    return True, capped_inputs, warnings_list

# Step 4: Robust Prediction Function
def predict_crop_robust(N, P, K, temperature, humidity, ph, rainfall, confidence_threshold=0.7):
    try:
        # Load the model
        with open('main_crop_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        # Validate and preprocess input
        is_valid, capped_inputs_or_error, warnings = validate_and_preprocess_input(
            N, P, K, temperature, humidity, ph, rainfall
        )
        if not is_valid:
            return {"error": capped_inputs_or_error, "prediction": None, "confidence": 0}
        
        # Prepare input as DataFrame to match training format
        input_df = pd.DataFrame([[
            capped_inputs_or_error['N'], capped_inputs_or_error['P'], capped_inputs_or_error['K'],
            capped_inputs_or_error['temperature'], capped_inputs_or_error['humidity'],
            capped_inputs_or_error['ph'], capped_inputs_or_error['rainfall']
        ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Predict crop and confidence
        predicted_crop = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        max_prob = float(max(probabilities))  # Convert np.float64 to float
        
        # Decision logic
        result = {
            "prediction": predicted_crop,
            "confidence": max_prob,
            "warnings": warnings if warnings else None
        }
        
        if max_prob < confidence_threshold:
            result["message"] = (f"Low confidence ({max_prob:.2f} < {confidence_threshold}). "
                                "Prediction may be unreliable. Consider a general crop like 'maize'.")
        
        return result
    
    except FileNotFoundError:
        return {"error": "Model file 'main_crop_model.pkl' not found", "prediction": None, "confidence": 0}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}", "prediction": None, "confidence": 0}

# Step 5: Test the Robust Model
if __name__ == "__main__":
    # Train the model (run once, comment out after)
    train_and_save_model()
    
    # Test cases
    test_cases = [
        [91,43,44,20,85,7,200],  # Normal input
        [500, 200, 300, 50.0, 10.0, 14.0, 1000.0],  # Extreme input
        [-10, 50, 60, 25.0, 75.0, 7.0, 150.0],  # Negative value
        [100, "invalid", 60, 25.0, 75.0, 7.0, 150.0]  # Invalid type
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_input}")
        result = predict_crop_robust(*test_input)
        print(f"Result: {result}")