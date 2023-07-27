import pandas as pd
import re
import numpy as np

# Load the CSV data
df = pd.read_csv('products.csv', sep=',')

# Function to extract content percentage
def extract_content(ingredient):
    if isinstance(ingredient, str):
        # Find matches for content in format like 'chocolat noir: 48%', 'cacao: 35%', etc
        match = re.search(r'([a-zA-Z\s]+)\s*:\s*([0-9]+)', ingredient, re.IGNORECASE)
        if match:
            # If a match is found, return the content percentage
            return match.group(2) + "%"
    # If no match is found or ingredient is not a string, return 'N/A'
    return 'N/A'

# Apply the function to the 'ingredients' column
df['ingredients'] = df['ingredients'].apply(extract_content)

# Convert 'N/A' to numpy's NaN
df['ingredients'] = df['ingredients'].replace('N/A', np.nan)

# Drop the rows where 'ingredients' column is NaN
df = df.dropna(subset=['ingredients'])

# Save the modified dataframe to a new CSV file
df.to_csv('modified_file.csv', sep='\t', index=False)
