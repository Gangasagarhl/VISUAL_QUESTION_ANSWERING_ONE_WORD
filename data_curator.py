import os
import subprocess
import pandas as pd
from tqdm import tqdm
from typing import Optional

class LlavaDataCurator:
    def __init__(
        self,
        metadata_csv: str = 'abo/metadata.csv',
        output_csv: str = 'curated_qa_dataset.csv',
        num_images: Optional[int] = None
    ):
        self.metadata_csv = metadata_csv
        self.output_csv = output_csv
        self.num_images = num_images
        self.prompt = """You are a visual question answering assistant.
Given this image, generate 3 diverse visual-only questions with single-word answers.
Only answer questions that can be answered just by looking at the image.
These are all the topic:
Object,Color,Position,Count,Action,Size,Shape,Emotion,Scene,Weather,Material,Text,Relation,Perspective,Activity,Place

On the above topics generate the query and then generate the one word answer to each question; 
Generate 9 different questions and answers logically relevant and not fixed for every image.
If the object is a device, ask about configuration kind of questions.
Format:
Q1: What is the shape of the object?
A1: <answer>
Q2: How many objects are there in the image?
A2: <answer>
Q3: What is the color of the object?
A3: <answer>
...
"""

    def load_metadata(self):
        print("Loading metadata...")
        df = pd.read_csv(self.metadata_csv)

        # Keep only rows whose image_path exists
        df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
        total = len(df)

        if self.num_images is not None:
            df = df.sample(n=self.num_images, random_state=42).reset_index(drop=True)
            print(f"Sampled {len(df)} valid images (out of {total}).")
        else:
            print(f"Using all {total} valid images.")

        self.sample_df = df

    def initialize_output(self):
        # Create or overwrite the CSV with the header row
        cols = ['image_id', 'question', 'answer']
        pd.DataFrame(columns=cols).to_csv(self.output_csv, index=False)
        print(f"Initialized output file: {self.output_csv}")

    def append_qa_to_csv(self, image_id: str, question: str, answer: str):
        # Append one row (image_id, question, answer)
        row = pd.DataFrame([{
            'image_id': image_id,
            'question': question,
            'answer': answer
        }])
        row.to_csv(self.output_csv, mode='a', header=False, index=False)

    def generate_qa_pairs(self):
        print("Starting QA pair generation (streaming to disk)...")
        for _, row in tqdm(self.sample_df.iterrows(), total=len(self.sample_df)):
            img_path = row['image_path']
            try:
                result = subprocess.run(
                    ['ollama', 'run', 'llava'],
                    input=f"{self.prompt}\n<image: {img_path}>\nCurrent Description: {row.get('description', '')}",
                    text=True,
                    capture_output=True,
                    timeout=120
                )

                response_lines = result.stdout.strip().splitlines()
                current_q = None

                for line in response_lines:
                    line = line.strip()
                    if line.startswith("Q") and ":" in line:
                        current_q = line.split(":", 1)[1].strip()
                    elif line.startswith("A") and ":" in line and current_q:
                        ans = line.split(":", 1)[1].strip()
                        # immediately write the pair
                        self.append_qa_to_csv(img_path, current_q, ans)
                        current_q = None

                # also append the product_type question (if present in metadata)
                if 'product_type' in row:
                    self.append_qa_to_csv(
                        img_path,
                        "What is the product type?",
                        row['product_type']
                    )

            except subprocess.TimeoutExpired:
                print(f"Timeout on image: {img_path}")
            except Exception as e:
                print(f"Error on image: {img_path} | {e}")

    def run(self):
        self.load_metadata()
        self.initialize_output()
        self.generate_qa_pairs()
        print("Done! All generated Q&A streamed to CSV.")

if __name__ == "__main__":
    curator = LlavaDataCurator(
        metadata_csv='merged_deduped_by_path.csv',
        output_csv='curated_qa_dataset.csv',
        num_images=2  # set to None to process *all* valid images
    )
    curator.run()
