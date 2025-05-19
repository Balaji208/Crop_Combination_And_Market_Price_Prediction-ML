import pandas as pd
import pickle
import os
from train import SubCropRecommender

def load_model(filename='subcrop_recommender.pkl'):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def evaluate_subcrop_recommender(model, test_data_path='merged_output.csv', num_recommendations=3):
    if not os.path.exists(test_data_path):
        print(f"Test data file '{test_data_path}' not found!")
        return
    
    test_df = pd.read_csv(test_data_path)
    required_cols = ['main-crop', 'sub-crop', 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    if not all(col in test_df.columns for col in required_cols):
        print("Missing required columns in test data!")
        return
    
    total_tests = 0
    correct_matches = 0
    mismatches = []
    
    for _, row in test_df.iterrows():
        expected_sub_crop = row['sub-crop']
        
        result = model.recommend_sub_crops(row['N'], row['P'], row['K'], 
                                           row['temperature'], row['humidity'], 
                                           row['ph'], row['rainfall'],
                                           num_recommendations)
        
        if "error" in result:
            print(f"Error for {row['main-crop']}: {result['error']}")
            continue
        
        predicted_sub_crops = [item['sub_crop'] for item in result['sub_crops']]
        total_tests += 1
        
        if expected_sub_crop in predicted_sub_crops:
            correct_matches += 1
        else:
            mismatches.append((row['main-crop'], expected_sub_crop, predicted_sub_crops))
    
    accuracy = (correct_matches / total_tests) * 100 if total_tests > 0 else 0.0
    
    print(f"SubCrop Recommender Accuracy: {accuracy:.2f}% (Correct: {correct_matches}/{total_tests})")
    
    with open("subcrop_mismatches.txt", "w") as f:
        for main_crop, expected, predicted in mismatches:
            f.write(f"Main Crop: {main_crop}, Expected: {expected}, Predicted: {predicted}\n")
    print("Mismatches logged in 'subcrop_mismatches.txt'")

if __name__ == "__main__":
    model = load_model()
    evaluate_subcrop_recommender(model)