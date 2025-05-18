import pandas as pd

# Load CSV
df = pd.read_csv("val.csv")

# Get unique IDs
unique_ids = df['image_id'].unique()

print(len(unique_ids))

half_ids = unique_ids[:len(unique_ids)//2]

reduced_df = df[df['image_id'].isin(half_ids)]

# # Save to new CSV
reduced_df.to_csv("val_red.csv", index=False)
