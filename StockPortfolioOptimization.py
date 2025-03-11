import pandas as pd
import os

# Define the path to the archive directory
archive_dir = 'archive'

# List all CSV files in the archive directory
csv_files = [f for f in os.listdir(archive_dir) if f.endswith('.csv')]

# List to store DataFrames
dataframes = []

# Read each CSV file into a DataFrame and append it to the list
for csv_file in csv_files:
    file_path = os.path.join(archive_dir, csv_file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Display the combined DataFrame
print(combined_df)