import pandas as pd
import pickle
import numpy as np

# Read the test data
test_data = pd.read_csv('test_data_RB.csv')
test_data.drop(columns = ['AP1', 'PB'], inplace=True)  # Drop AP1 and PB columns if they exist
# Prepare the data by dropping unnecessary columns
player_names = test_data['Player Name'].copy()  # Save player names
test_data = test_data.drop(columns=[col for col in test_data.columns if 'St' in col or 'Year' in col or col == 'Player'])
model_files = ['model_2011_2021_St2+.pkl', 'model_2011_2020_St3+.pkl', 
               'model_2011_2019_St4+.pkl','model_2011_2018_St5+.pkl',
               'model_AP1_2011_2020.pkl','model_PB_2011_2020.pkl']
for model_file in model_files:
    # Load the model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)['model']
    # Create a copy of test_data to avoid modifying the original
    prediction_data = test_data.copy()
    # Drop 'PB' column for all models except model_PB_2011_2020.pkl
    
    # Assume we need to predict probabilities using all columns except certain ID or target columns
    features = prediction_data.select_dtypes(include=[np.number]).columns.tolist()

    # Generate predictions
    predictions = model.predict_proba(prediction_data[features])

    # If this is a binary classification, take the probability of the positive class
    if predictions.shape[1] == 2:
        predictions = predictions[:, 1]

    # Save or display the results
    results = pd.DataFrame({
        'ID': test_data.index,  # Assuming index is the ID, adjust if needed
        'predicted_probability': predictions
    })
    
    # Store results from this model with a column name based on the model file
    model_name = model_file.split('.')[0]  # Remove file extension
    if 'all_results' not in locals():
        # First time through the loop - create DataFrame with ID column and Player names
        all_results = pd.DataFrame({'ID': test_data.index, 'Player': player_names})
    
    # Add predictions from this model as a new column
    all_results[f'{model_name}_probability'] = predictions
    
    print(f"Processed {model_file}")
    print(results.head())
all_results.drop(columns=['ID'], inplace=True)  # Drop ID column if not needed
# After the loop, save the combined results
print("\nCombined results:")
print(all_results.head())
all_results.to_csv('combined_prediction_results.csv', index=False)