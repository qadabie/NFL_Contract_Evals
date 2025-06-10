import pandas as pd
from Add_Data import add_player_data

# Load the parquet file
file_path = 'nfl_draft_data.parquet'
df = pd.read_parquet(file_path)

position = 'Running Back'
year1 = 2012
year2 = 2025

# # Display the unique values of 'year' and 'position' columns
# print("Unique years in the dataframe:")
# print(df['year'].unique())
# print("\nUnique positions in the dataframe:")
# print(df['position'].unique())

# # Display counts of each value
# print("\nCount of records by year:")
# print(df['year'].value_counts().sort_index())
# print("\nCount of records by position:")
# print(df['position'].value_counts())
df = df[(df['position'] == position) & (df['year'] >= year1) & (df['year'] <= year2)]
df= df[['name','pros','cons']]
# Apply add_player_data function to update 'pros' column with player strengths
updated_names = {'Ronald Jones II': 'Ronald Jones',
                 'Devon Achane': "De'Von Achane",
                 'Audric Estime': 'Audric Estimé'}
df = df[df['name'].isin(updated_names.keys())]
df['name'] = df['name'].replace(updated_names)
#df = df.apply(lambda row: add_player_data(row['name'], 'Strengths', row['pros']) or row, axis=1)
df = df.apply(lambda row: add_player_data(row['name'], 'Weaknesses', row['cons']) or row, axis=1)