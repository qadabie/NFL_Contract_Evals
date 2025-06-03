import pandas as pd
import os
from plotly.offline import plot

import plotly.express as px
import plotly.io as pio

# Set Plotly to render in notebook or standalone HTML
pio.renderers.default = "browser"

# Path to the folder containing the draft data files
folder_path = r"C:\Users\adabi\OneDrive\Documents\NFL_Contract_Evals\draft_data"

# Set the threshold value for player starter status
st_threshold = 3  # Changed from 4 to 3, now as a variable

# Initialize an empty list to store dataframes
dfs = []

# Loop through each file in the folder
for idx, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(('.xlsx', '.csv')):  # Handle both Excel and CSV files
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Determine file type and read accordingly
            if filename.endswith('.xlsx'):
                # First file doesn't skip rows, others skip 1
                skiprows = 1 if idx > 0 else 0
                df = pd.read_excel(file_path, skiprows=skiprows)
            else:  # CSV file
                df = pd.read_csv(file_path, skiprows=0)
            
            # Add to our list
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Combine all dataframes
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    # Filter out positions 'K' and 'P'
    combined_df = combined_df[~combined_df['Pos'].isin(['K', 'P','FB'])]

    # Combine positions
    combined_df['Pos'] = combined_df['Pos'].replace({
        'G': 'OL', 'T': 'OL', 'C': 'OL',
        'NT': 'DL', 'DT': 'DL',
        'CB': 'DB', 'S': 'DB'
    })
    # Create a new column for players with St >= threshold
    combined_df['High_St'] = combined_df['St'] >= st_threshold
    
    # Group by round and position, calculate percentage of players with St >= threshold
    grouped = combined_df.groupby(['Rnd', 'Pos']).agg(
        total_players=('High_St', 'count'),
        high_st_players=('High_St', 'sum')
    ).reset_index()
    
    grouped['percentage'] = (grouped['high_st_players'] / grouped['total_players']) * 100
    
    # Save the processed data to a CSV file
    processed_data_path = os.path.join(folder_path, 'processed_draft_data.csv')
    combined_df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")
else:
    print("No data files found in the specified folder.")
