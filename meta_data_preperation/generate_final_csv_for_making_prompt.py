import pandas as pd

# Load your DataFrame (replace with actual loading code if needed)
# Example: data2 = pd.read_csv("your_file.csv")
data1 =  pd.read_csv("../summed_json__flat.csv")
data2 =  pd.read_csv("../merged_images_listings.csv")
data3 =  pd.read_csv("../images.csv")

#changing the name of the column
data1.rename(columns={'main_image_id': "image_id"}, inplace=True)

# Ensure all necessary columns exist
required_columns = ["image_id", "height", "width", "path", "asin", "brand", "color", "product_type"]
data2 = data2[required_columns]

# Convert target columns to string type to prevent type issues
for col in ["asin", "brand", "color", "product_type"]:
    data2[col] = data2[col].astype(str)

# Group by image_id and aggregate with optimized functions
merged_df2 = data2.groupby("image_id").agg({
    "height": "first",            # assuming height is the same for all duplicates
    "width": "first",             # assuming width is the same for all duplicates
    "path": "first",              # assuming path is the same for all duplicates
    "asin": lambda x: ", ".join(set(x)),
    "brand": lambda x: ", ".join(set(x)),
    "color": lambda x: ", ".join(set(x)),
    "product_type": lambda x: ", ".join(set(x)),
}).reset_index()

# Show original and merged shape
print("Original shape:", data2.shape)
print("Grouped shape:", merged_df2.shape)

# Optional: Save to CSV
# merged_df.to_csv("merged_output.csv", index=False)

final_df = pd.merge(merged_df2, data3, how='left', on='image_id')
print("merged_df shape:", merged_df2.shape)
print("data3 shape:", data3.shape)
print("final_df shape:", final_df.shape)

final_df = pd.merge(final_df, data3, how='left', on='image_id')
print(final_df.shape,data1.shape)
final_csv = pd.merge(data1, final_df,how='left',on='image_id')
print(final_csv.shape)

final_csv['path']="abo-images-small/images/small/"+final_csv['path_y']
final_csv.to_csv("../final_ready_to_go.csv",index=False)