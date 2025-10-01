

import pandas as pd
import os

# Path to your original 28-class CSV
ORIGINAL_CSV = "data/combined_dataset/annotations.csv"

# Path to save the new filtered CSV
FILTERED_CSV = "data/Dataset_3_classes/annotations_3.csv"

# Classes you want to keep
SELECTED_CLASSES = ["healthy", "dermatitis", "demodicosis"]

# Read original CSV
df = pd.read_csv(ORIGINAL_CSV)

# Filter rows
df_filtered = df[df['label'].isin(SELECTED_CLASSES)].reset_index(drop=True)

# Save filtered CSV
os.makedirs(os.path.dirname(FILTERED_CSV), exist_ok=True)
df_filtered.to_csv(FILTERED_CSV, index=False)

print(f"✅ Filtered CSV saved with {len(df_filtered)} rows at {FILTERED_CSV}")
