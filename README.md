Test read me file

# Importing the necessary Python libraries
import pandas as pd
import numpy as np

# Convert the 'point_differential' column to numeric
df['point_differential'] = pd.to_numeric(df['point_differential'], errors='coerce')

# Add the new column 'Outcome' based on the condition
df['Outcome'] = df['point_differential'].apply(lambda x: 'W' if x < 0 else 'L')

# Find the index of the point_differential column
col_index = df.columns.get_loc('point_differential') + 1

# Move the Outcome column to after the point_differential column
df.insert(col_index, 'Outcome', df.pop('Outcome'))

df['Outcome'].value_counts()
Outcome
W    4103
L    3692
Name: count, dtype: int64

nan_count = df['Outcome'].isna().sum()
print("NaN count:", nan_count)
NaN count: 0

Lets see if I got this working!

