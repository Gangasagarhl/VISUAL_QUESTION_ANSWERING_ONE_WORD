import pandas as pd
import os
import re

folder_path = "BLIP2_after_finetune"
output_file_name = "blip2_finetuned_evaluation.csv"
rexp = r'evaluation_(\d+)_\d+\.csv'

csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
csv_files.sort(key=lambda x: int(re.search(rexp, x).group(1)))

combined_df = pd.concat([pd.read_csv(os.path.join(folder_path, f)) for f in csv_files], ignore_index=True)
combined_df.to_csv(os.path.join("Data",output_file_name), index=False)

print(f"Combined {len(csv_files)} files. Final shape: {combined_df.shape}")
