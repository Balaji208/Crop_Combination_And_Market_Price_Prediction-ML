def calculate_subcrop_accuracy(self, num_recommendations=3):
    total_tests = 0
    correct_matches = 0
    skipped_datasets = []
    
    with open('subcrop_accuracy_debug.txt', 'w') as debug_file:
        for main_crop, filename in self.crop_name_mapping.items():
            file_path = os.path.join(self.subcrop_dir, filename)
            if not os.path.exists(file_path):
                skipped_datasets.append(f"{main_crop}: File {filename} not found")
                debug_file.write(f"Skipping {main_crop}: {filename} not found\n")
                continue
                
            try:
                sub_crop_df = pd.read_csv(file_path)
                required_cols = ['sub-crop', 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
                if not all(col in sub_crop_df.columns for col in required_cols):
                    skipped_datasets.append(f"{main_crop}: Missing required columns")
                    debug_file.write(f"Skipping {main_crop}: {filename} missing required columns\n")
                    continue
                    
                if len(sub_crop_df) < 50 or len(sub_crop_df['sub-crop'].unique()) < 3:
                    skipped_datasets.append(f"{main_crop}: Insufficient samples ({len(sub_crop_df)}) or unique sub-crops ({len(sub_crop_df['sub-crop'].unique())})")
                    debug_file.write(f"Skipping {main_crop}: Insufficient samples or classes\n")
                    continue
                    
                for _, row in sub_crop_df.iterrows():
                    expected_sub_crop = row['sub-crop']
                    test_input = [row['N'], row['P'], row['K'], row['temperature'], 
                                  row['humidity'], row['ph'], row['rainfall']]
                    
                    result = self.recommend_sub_crops(*test_input, num_recommendations=num_recommendations)
                    
                    if "error" in result:
                        skipped_datasets.append(f"{main_crop}: Recommendation error - {result['error']}")
                        debug_file.write(f"Error for {main_crop}: {result['error']}\n")
                        continue
                    
                    predicted_sub_crops = [item['sub_crop'] for item in result['sub_crops']]
                    total_tests += 1
                    
                    if expected_sub_crop in predicted_sub_crops:
                        correct_matches += 1
                    else:
                        debug_file.write(f"Mismatch for {main_crop}: Expected {expected_sub_crop}, Got {predicted_sub_crops}\n")
            
            except Exception as e:
                skipped_datasets.append(f"{main_crop}: Data loading error - {str(e)}")
                debug_file.write(f"Error loading {main_crop}: {str(e)}\n")
                continue
            
        accuracy = (correct_matches / total_tests) * 100 if total_tests > 0 else 0.0
        accuracy_message = f"Accuracy: {accuracy:.2f}% (Correct: {correct_matches}/{total_tests})"
        debug_file.write(f"\n{accuracy_message}\n")
    
    return accuracy, accuracy_message