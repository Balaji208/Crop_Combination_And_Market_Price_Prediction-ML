from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import os

app = Flask(__name__)

# Global variables for models
main_model = None
sub_model = None

# Realistic Ranges
realistic_ranges = {
    'N': (0, 200), 'P': (0, 200), 'K': (0, 250), 'temperature': (5, 50),
    'humidity': (0, 100), 'ph': (3, 11), 'rainfall': (0, 500)
}

# Crop Mapping
crop_name_mapping = {
    'Rice': 'Rice_subcrop_data.csv',
    'Maize': 'Maize_subcrop_data.csv',
    'Bengal Gram (Gram)(Whole)': 'Bengal Gram (Gram)(Whole)_subcrop_data.csv',
    'Pegeon Pea (Arhar Fali)': 'Pegeon Pea (Arhar Fali)_subcrop_data.csv',
    'Moath Dal': 'Moath Dal_subcrop_data.csv',
    'Green Gram (Moong)(Whole)': 'Green Gram (Moong)(Whole)_subcrop_data.csv',
    'Black Gram (Urd Beans)(Whole)': 'Black Gram (Urd Beans)(Whole)_subcrop_data.csv',
    'Lentil (Masur)(Whole)': 'Lentil (Masur)(Whole)_subcrop_data.csv',
    'Pomegranate': 'Pomegranate_subcrop_data.csv',
    'Banana': 'Banana_subcrop_data.csv',
    'Mango': 'Mango_subcrop_data.csv',
    'Grapes': 'Grapes_subcrop_data.csv',
    'Water Melon': 'Water Melon_subcrop_data.csv',
    'Karbuja (Musk Melon)': 'Karbuja (Musk Melon)_subcrop_data.csv',
    'Apple': 'Apple_subcrop_data.csv',
    'Orange': 'Orange_subcrop_data.csv',
    'Papaya': 'Papaya_subcrop_data.csv',
    'Coconut': 'Coconut_subcrop_data.csv',
    'Cotton': 'Cotton_subcrop_data.csv',
    'Jute': 'Jute_subcrop_data.csv',
    'Coffee': 'Coffee_subcrop_data.csv'
}

# SubCropRecommender Class
class SubCropRecommender:
    def __init__(self, subcrop_dir='../datasets/sub_crop_data/'):
        self.subcrop_dir = subcrop_dir
        self.crop_name_mapping = crop_name_mapping
        self.realistic_ranges = realistic_ranges

    def validate_and_preprocess_input(self, N, P, K, temperature, humidity, ph, rainfall):
        inputs = {'N': N, 'P': P, 'K': K, 'temperature': temperature, 
                  'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
        for param, val in inputs.items():
            try:
                inputs[param] = float(val)
            except (ValueError, TypeError):
                return False, f"Invalid input: {param} must be a number", []
        capped_inputs = {}
        warnings_list = []
        for param, val in inputs.items():
            min_val, max_val = self.realistic_ranges[param]
            if val < min_val or val > max_val:
                warnings_list.append(f"{param} ({val}) outside realistic range ({min_val}-{max_val}), capped")
                capped_inputs[param] = max(min_val, min(val, max_val))
            else:
                capped_inputs[param] = val
        return True, capped_inputs, warnings_list

    def recommend_sub_crops(self, main_crop, N, P, K, temperature, humidity, ph, rainfall, num_recommendations=3):
        try:
            is_valid, capped_inputs, warnings = self.validate_and_preprocess_input(
                N, P, K, temperature, humidity, ph, rainfall
            )
            if not is_valid:
                return {"error": capped_inputs, "sub_crops": [], "warnings": warnings}
            
            if main_crop not in self.crop_name_mapping:
                return {"error": f"No sub-crop mapping for {main_crop}", "sub_crops": [], "warnings": warnings}
            
            subcrop_filename = self.crop_name_mapping[main_crop]
            subcrop_file = os.path.join(self.subcrop_dir, subcrop_filename)
            
            if not os.path.exists(subcrop_file):
                return {"error": f"Sub-crop file {subcrop_filename} not found", "sub_crops": [], "warnings": warnings}
            
            sub_crop_df = pd.read_csv(subcrop_file)
            required_cols = ['sub-crop', 'N', 'P', 'K', 'temperature', 'rainfall', 'ph', 'humidity']
            missing_cols = [col for col in required_cols if col not in sub_crop_df.columns]
            if missing_cols:
                return {"error": f"Missing columns: {missing_cols}", "sub_crops": [], "warnings": warnings}
            
            input_vector = np.array([[capped_inputs['N'], capped_inputs['P'], capped_inputs['K'], 
                                      capped_inputs['temperature'], capped_inputs['rainfall'], 
                                      capped_inputs['ph'], capped_inputs['humidity']]])
            sub_crop_features = sub_crop_df[['N', 'P', 'K', 'temperature', 'rainfall', 'ph', 'humidity']].values
            
            distances = euclidean_distances(input_vector, sub_crop_features)[0]
            sub_crops_with_distances = list(zip(sub_crop_df['sub-crop'], distances))
            sorted_sub_crops = sorted(sub_crops_with_distances, key=lambda x: x[1])[:num_recommendations]
            recommended_sub_crops = [{"sub_crop": crop, "distance": float(dist)} for crop, dist in sorted_sub_crops]
            
            return {"sub_crops": recommended_sub_crops, "warnings": warnings if warnings else None}
        except Exception as e:
            return {"error": str(e), "sub_crops": [], "warnings": None}

# Load models functions
def load_main_crop_model(filename='main_crop_model.pkl'):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def load_subcrop_model(filename='subcrop_recommender.pkl'):  # Match the filename you pickled
    with open(filename, 'rb') as file:
        return pickle.load(file)

def predict_main_crop(N, P, K, temperature, humidity, ph, rainfall, model):
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    main_crop = model.predict(input_df)[0]
    confidence = float(max(model.predict_proba(input_df)[0]))
    return {"main_crop": main_crop, "confidence": confidence}

# Load models at startup
def init_models():
    global main_model, sub_model
    try:
        main_model = load_main_crop_model('main_crop_model.pkl')
        sub_model = load_subcrop_model('subcrop_recommender.pkl')  # Match the filename
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

init_models()

@app.route('/predict_main_crop', methods=['POST'])
def predict_main():
    data = request.get_json()
    try:
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        result = predict_main_crop(N, P, K, temperature, humidity, ph, rainfall, main_model)
        return jsonify(result)
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

@app.route('/recommend_sub_crops', methods=['POST'])
def recommend_sub():
    data = request.get_json()
    try:
        main_crop = data['main_crop']
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        result = sub_model.recommend_sub_crops(main_crop, N, P, K, temperature, humidity, ph, rainfall)
        return jsonify(result)
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)