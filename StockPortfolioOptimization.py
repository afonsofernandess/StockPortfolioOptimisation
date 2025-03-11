import pandas as pd
import glob

# Get all CSV file paths
csv_files = glob.glob("archive/*.csv")


df_list = [pd.read_csv(csv_files[0])]

# Read the rest without headers
df_list += [pd.read_csv(file, header=None) for file in csv_files[1:]]

# Assign column names from the first file to all
for df in df_list[1:]:
    df.columns = df_list[0].columns

# Concatenate all DataFrames
final_df = pd.concat(df_list, ignore_index=True)

print(final_df)