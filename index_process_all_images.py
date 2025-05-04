# process_all_images.py

import pandas as pd
from vqa_pipeline import VQAPipeline
from write_into_csv import CSVWriter

INPUT_CSV   = "images_list.csv"   
OUTPUT_CSV  = "output.csv"       

def main():
    # 1) Read all image paths
    df = pd.read_csv(INPUT_CSV)
    paths = df["image_path"].dropna().tolist()

    # 2) Instantiate pipeline and CSV writer
    pipeline = VQAPipeline()
    writer   = CSVWriter(OUTPUT_CSV)

    # 3) For each image, run VQA and write results
    for img_path in paths:
        try:
            result = pipeline.run(image_path=img_path, num_questions=5)
            writer.write(result)
            print(f"Processed {img_path}")
        except Exception as e:
            print(f"Error on {img_path}: {e}")

if __name__ == "__main__":
    main()
