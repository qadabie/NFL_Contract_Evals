import pandas as pd
df = pd.read_csv(r"C:\Users\adabi\OneDrive\Documents\NFL_Contract_Evals\processed_RB_data.csv")
# Display row 6740 (note: if index is 0-based, this is the 6741st row)
# Count missing values in each column
missing_values = df.isnull().sum()
print("Missing values by column:")
print(missing_values)

# Optional: Show as percentage
missing_percentage = (missing_values / len(df)) * 100
print("\nMissing values percentage by column:")
print(missing_percentage.round(2))

# Display row at index 25
print("\nRow at index 25:")
print(df.iloc[24])
# Check if the column exists
if 'processed_combined_text' in df.columns:
    # Get the first non-null value
    sample_row = df['processed_combined_text'].dropna().iloc[0]
    # Extract first sentence (assuming sentences end with periods)
    first_sentence = sample_row.split('.')[0] + '.' if '.' in sample_row else sample_row
    print("\nFirst sentence in processed_combined_text:")
    print(first_sentence)
else:
    print("\nColumn 'processed_combined_text' not found in the dataframe")