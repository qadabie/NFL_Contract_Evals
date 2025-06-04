import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
from imblearn.over_sampling import SMOTE

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Read the CSV file
df = pd.read_csv('processed_RB_data.csv')
# Delete columns not needed
df = df.drop(['Link','Analysis','Strengths','Weaknesses','College','Position','Team','combined_text', 'processed_combined_text'], axis=1)

# Read the draft data
draft_df = pd.read_csv(r"C:\Users\adabi\OneDrive\Documents\NFL_Contract_Evals\draft_data\processed_draft_data.csv")

# Merge the St column based on player name
df = pd.merge(df, draft_df[['Player','St','PB','AP1']], left_on='Player Name', right_on='Player', how='left')

# Replace NaN values in 'St' with 0 or another appropriate value
df['St'] = df['St'].fillna(0)
df = df[df['Pick'] != 'Undrafted']  # Ensure no undrafted players are included
df = df[df['Pick'] < 160]  # Filter out picks greater than or equal to 160
quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=42)
# Create a list of columns to transform (all except excluded ones)
columns_to_transform = [col for col in df.columns if col not in ['Player Name', 'St', 'Pick', 'Player', 'Year','PB','AP1']]

# Apply the quantile transformer to these columns
df[columns_to_transform] = quantile_transformer.fit_transform(df[columns_to_transform])
# Convert Pick column to numeric (in case it's not already)
df['Pick'] = pd.to_numeric(df['Pick'])

# Apply standard scaling to the Pick column
scaler_pick = StandardScaler()
df['Pick'] = scaler_pick.fit_transform(df[['Pick']])
test_data = df[df['Year'] > 2021].copy()
# Save the test data to a CSV file
test_data.to_csv('test_data_RB.csv', index=False)
print("Test data saved to 'test_data_RB.csv'")
# Display the first 5 rows of the test data
print(test_data.head(1))
df.drop(['AP1'], axis=1, inplace=True, errors='ignore')  # Drop AP1 column if it exists
# Remove duplicate rows
df = df.drop_duplicates()
# Keep only the first occurrence of each Player Name
df = df.drop_duplicates(subset=['Player Name'], keep='first')
print(f"Number of rows after removing duplicates: {len(df)}")
df.to_csv('scaled_RB_data.csv', index=False)
# Define the model configurations
model_configs = [
    {'years': (2011, 2021), 'st_threshold': 2, 'name': 'model_2011_2021_St2+'},
    {'years': (2011, 2020), 'st_threshold': 3, 'name': 'model_2011_2020_St3+'},
    {'years': (2011, 2019), 'st_threshold': 4, 'name': 'model_2011_2019_St4+'},
    {'years': (2011, 2018), 'st_threshold': 5, 'name': 'model_2011_2018_St5+'}
]
# Define the new model for PB prediction
print("\n--- Building PB Prediction Model ---")

# Filter data by year range (2011-2020)
PB_filtered_df = df[(df['Year'] >= 2011) & (df['Year'] <= 2020)]

# Create target variable based on PB threshold (1 or more)
PB_filtered_df['High_PB'] = (PB_filtered_df['PB'] >= 1).astype(int)

print(f"Number of players from 2011-2020: {len(PB_filtered_df)}")
print(f"Players with PB >= 1: {PB_filtered_df['High_PB'].sum()}")
print(f"Players with PB < 1: {len(PB_filtered_df) - PB_filtered_df['High_PB'].sum()}")

# Prepare features (X) and target (y)
y_PB = PB_filtered_df['High_PB']
X_PB = PB_filtered_df.drop(['Player', 'High_PB', 'Year', 'St', 'PB'], axis=1)

# Store player names for reference but remove from features
player_names_PB = X_PB['Player Name']
X_PB = X_PB.drop(['Player Name'], axis=1)

# Split the data
X_train_PB, X_test_PB, y_train_PB, y_test_PB = train_test_split(
    X_PB, y_PB, test_size=0.3, random_state=42, stratify=y_PB,
)

# Create and train a logistic regression model
PB_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
PB_model.fit(X_train_PB, y_train_PB)

# Evaluate the model
y_pred_PB = PB_model.predict(X_test_PB)
accuracy_PB = accuracy_score(y_test_PB, y_pred_PB)
print(f"Test Accuracy: {accuracy_PB:.4f}")
print("Classification Report:")
print(classification_report(y_test_PB, y_pred_PB))

# Get feature importance
feature_importance_PB = pd.DataFrame({
    'Feature': X_PB.columns,
    'Importance': np.abs(PB_model.coef_[0])
}).sort_values('Importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance_PB.head(5))

# Save the PB model
PB_model_data = {
    'model': PB_model,
    'scaler': quantile_transformer,
    'feature_names': list(X_PB.columns),
    'PB_threshold': 1,
    'year_range': (2011, 2020)
}

# Save the model to a pickle file
with open('model_PB_2011_2020.pkl', 'wb') as f:
    pickle.dump(PB_model_data, f)
print("PB prediction model saved as 'model_PB_2011_2020.pkl'")
for config in model_configs:
    print(f"\n--- Building {config['name']} ---")
    
    # Filter data by year range
    year_min, year_max = config['years']
    filtered_df = df[(df['Year'] >= year_min) & (df['Year'] <= year_max)]
    
    # Create target variable based on St threshold
    threshold = config['st_threshold']
    filtered_df['High_St'] = (filtered_df['St'] >= threshold).astype(int)
    
    print(f"Number of players from {year_min}-{year_max}: {len(filtered_df)}")
    print(f"Players with St >= {threshold}: {filtered_df['High_St'].sum()}")
    print(f"Players with St < {threshold}: {len(filtered_df) - filtered_df['High_St'].sum()}")
    
    # Prepare features (X) and target (y)
    y = filtered_df['High_St']
    X = filtered_df.drop(['Player', 'High_St', 'Year', 'St','PB'], axis=1)
    
    # Store player names for reference but remove from features
    player_names = X['Player Name']
    X = X.drop(['Player Name'], axis=1)
    
    # Standardize features
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )
    # Import SMOTE for handling class imbalance

    # Apply SMOTE to balance classes
    #print("Applying SMOTE to balance classes...")
    # Count samples in minority class
    # min_class_count = min(pd.Series(y_train).value_counts())
    # # Set k_neighbors to be at most the number of samples in minority class
    # k_neighbors = min(5, min_class_count - 1)
    # if k_neighbors > 0:
    #     smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    #     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    # else:
    #     print("Not enough samples for SMOTE. Using original training data.")
    #     X_train_resampled, y_train_resampled = X_train, y_train

    # print(f"Original training data shape: {X_train.shape}")
    # print(f"Resampled training data shape: {X_train_resampled.shape}")
    # print(f"Original class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    # print(f"Resampled class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")

    # Use the resampled data for training
    # X_train = X_train
    # y_train = y_train_resampled
    # Create and train a logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(model.coef_[0])
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head(5))
    
    # Save the model, scaler, and feature names
    model_data = {
        'model': model,
        'scaler': quantile_transformer,
        'feature_names': list(X.columns),
        'st_threshold': threshold,
        'year_range': config['years']
    }
with open('model_2011_2021_St3+.pkl', 'rb') as f:
    model_data = pickle.load(f)
    prediction_model = model_data['model']
    feature_cols = model_data.get('feature_names', [])

    if test_data is None or len(test_data) == 0:
        print("No 2025 data found. Creating sample prediction data...")
    # Get feature columns for prediction based on the model's required features
    
    # Make predictions
    if len(feature_cols) > 0:
            try:
                # Save Player Name to remerge later
                player_names = test_data['Player Name'].copy()
                
                # Prepare features (excluding non-feature columns)
                X_pred = test_data.drop(['Player Name','Player','St','Year','PB','AP1'], axis=1, errors='ignore')
                
                # Standardize features using the saved scaler
                if isinstance(prediction_model, dict):
                    scaler = prediction_model.get('scaler')
                    model = prediction_model.get('model')
                else:
                    with open('model_2011_2020_St3+.pkl', 'rb') as f:
                        model_data = pickle.load(f)
                        scaler = model_data.get('scaler')
                        model = prediction_model
                
                # Transform the features
                X_pred_scaled = X_pred.copy()
                
                # Standardize features using the saved scaler
                #scaler = prediction_model['scaler'] if isinstance(prediction_model, dict) else pickle.load(open('model_2011_2020_St3+.pkl', 'rb'))['scaler']
                #X_pred_scaled = scaler.transform(X_pred)
                
                # Make predictions
                model = prediction_model['model'] if isinstance(prediction_model, dict) else prediction_model
                # Get probability of positive class (second column of predict_proba output)
                predictions_2025 = model.predict_proba(X_pred_scaled)[:, 1]
                
                # Create a new DataFrame with player names and predictions
                results_df = pd.DataFrame({
                    'Player Name': player_names,
                    'predicted_value': predictions_2025
                })
                
                # Merge back with original data if needed
                test_data = pd.merge(results_df, test_data, on='Player Name', how='left')
                test_data['predicted_value'] = predictions_2025
                
                # Display the predictions
                print("\n2025 Predictions:")
                print(test_data[['Player Name', 'predicted_value']].head(10) if 'Player Name' in test_data.columns else test_data['predicted_value'].head(10))
                
                # Save predictions to CSV
                test_data.to_csv('RB_predictions_2025.csv', index=False)
                print("Predictions saved to 'RB_predictions_2025.csv'")
            except Exception as e:
                print(f"Error making predictions: {e}")
    else:
            print("No suitable feature columns found for prediction")