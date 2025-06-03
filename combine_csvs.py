import pandas as pd

year = 2008
# Load the CSV file into a pandas DataFrame
# Initialize an empty list to store DataFrames
dataframes = []

# Iterate over numbers 0 to 10
for i in range(18):
    # Construct the file name with the current iteration
    file_name = f'WRs_drafted/nfl-2025-05-27 ({i}).csv'
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_name)
    
    # Display the head of the DataFrame with all columns
    pd.set_option('display.max_columns', None)
    
    # Rename columns to 'col X' where X is an ascending number
    df.columns = [f'col {j}' for j in range(len(df.columns))]
    
    # Combine columns 'col 4' and 'col 9', 'col 5' and 'col 10', etc.
    for col1, col2 in zip([f'col {j}' for j in range(4, 9)], [f'col {j}' for j in range(9, 14)]):
        if col1 in ['col 7', 'col 8'] or col2 in ['col 12', 'col 13']:
            continue
        df[col1] = df[col1].fillna('') + df[col2].fillna('')
    
    # Drop the now redundant columns 'col 7', 'col 8', and 'col 9' to 'col 13'
    df = df.drop(columns=['col 0', 'col 7', 'col 8'] + [f'col {j}' for j in range(9, 14)])
    
    # Ensure columns 'col 8' and 'col 13' are float
    if 'col 8' in df.columns:
        df['col 8'] = pd.to_numeric(df['col 8'], errors='coerce')
    if 'col 13' in df.columns:
        df['col 13'] = pd.to_numeric(df['col 13'], errors='coerce')
    
    # Drop the now redundant columns 'col 9' to 'col 13'
    df = df.drop(columns=[f'col {j}' for j in range(9, 14) if f'col {j}' in df.columns])
    
    # Add the current year to the DataFrame
    df['year'] = year + i
    # if i >= 11:
    #     df['year'] = year + i - 15
    
    # Process 'col 4' to replace its values
    df['col 4'] = df['col 4'].apply(
        lambda x: (int(x.split(',')[0].split()[1]) - 1) * 32 + int(x.split(',')[1].split()[1]) 
        if isinstance(x, str) and "Rnd" in x and "Pick" in x else x
    )
    #print(df.head(1))  # Display the first row of the DataFrame
    # Rename columns
    df.columns = ['Link', 'Player', 'College', 'Pick', 'Position', 'Team', 'Year']
    #print(df.tail(1))
    # Append the processed DataFrame to the list
    dataframes.append(df)
# Combine all DataFrames into one total DataFrame
total_df = pd.concat(dataframes, ignore_index=True)
total_df = total_df[total_df["Pick"] != "Undrafted"]  # Filter out undrafted players
# Save the combined DataFrame to a new CSV file
total_df.to_csv('drafted_wrs_combined.csv', index=False)