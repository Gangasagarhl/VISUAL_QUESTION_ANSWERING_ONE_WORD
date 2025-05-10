import json
import os
import pandas as pd

# Folder containing JSON files
input_folder = "../../abo-listings/listings/metadata/listings/"
output_csv = "../summed_json__flat.csv"

def extract_value(field):
    if isinstance(field, list) and len(field) > 0:
        return field[0].get("value", "")
    return ""

def extract_lang_value(field):
    if isinstance(field, list) and len(field) > 0:
        return field[0].get("value", "")
    return ""

def extract_bullet_points(field):
    """Concatenate bullet points where language_tag starts with 'en'."""
    if isinstance(field, list):
        return " ".join([
            b.get("value", "")
            for b in field
            if isinstance(b, dict) and
               b.get("language_tag", "").startswith("en") and
               "value" in b
        ])
    return ""

flattened_data = []

for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        row = {
                            "item_id": obj.get("item_id", ""),
                            "brand": extract_lang_value(obj.get("brand", "")),
                            "item_name": extract_lang_value(obj.get("item_name", "")),
                            "color": extract_lang_value(obj.get("color", "")),
                            "product_type": extract_value(obj.get("product_type", "")),
                            "style": extract_lang_value(obj.get("style", "")),
                            "main_image_id": obj.get("main_image_id", ""),
                            "country": obj.get("country", ""),
                            "marketplace": obj.get("marketplace", ""),
                            "domain_name": obj.get("domain_name", ""),
                            "node_name": obj.get("node", [{}])[0].get("node_name", ""),
                            "description": extract_bullet_points(obj.get("bullet_point", ""))
                        }
                        flattened_data.append(row)
                    except json.JSONDecodeError as e:
                        print(f"Skipping malformed line in {filename}: {e}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

df = pd.DataFrame(flattened_data)
df.to_csv(output_csv, index=False)
print(f"Saved metadata CSV to: {output_csv}")
