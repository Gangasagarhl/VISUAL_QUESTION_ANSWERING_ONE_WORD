import os
import pandas as pd

class CSVWriter:
    """Class to handle writing image data with questions and answers to a CSV."""

    def __init__(self, output_csv_path: str):
        self.output_csv_path = output_csv_path

    def write(self, data: dict):
        """
        Write or append a structured record to CSV.
        data format:
        {
            "image_path": str,
            "caption": str,
            "questions": List[str],
            "answers": List[str]
        }
        """
        image_path = data['image_path']
        caption = data['caption']
        questions = data['questions']
        answers = data['answers']

        # Prepare rows for each question-answer pair
        rows = [
            {
                'image_path': image_path,
                'question': q,
                'answer': a,
                'caption': caption
            }
            for q, a in zip(questions, answers)
        ]

        df_new = pd.DataFrame(rows)

        # Append if file exists, else create with header
        if os.path.exists(self.output_csv_path):
            df_new.to_csv(self.output_csv_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(self.output_csv_path, mode='w', header=True, index=False)


import os
import pandas as pd

class CSVWriter:
    """Class to handle writing image data with questions and answers to a CSV."""

    def __init__(self, output_csv_path: str):
        self.output_csv_path = output_csv_path

    def write(self, data: dict):
        """
        Write or append a structured record to CSV.
        data format:
        {
            "image_path": str,
            "caption": str,
            "questions": List[str],
            "answers": List[str]
        }
        """
        image_path = data['image_path']
        caption = data['caption']
        questions = data['questions']
        answers = data['answers']

        # Prepare rows for each question-answer pair
        rows = [
            {
                'image_path': image_path,
                'question': q,
                'answer': a,
                'caption': caption
            }
            for q, a in zip(questions, answers)
        ]

        df_new = pd.DataFrame(rows)

        # Append if file exists, else create with header
        if os.path.exists(self.output_csv_path):
            df_new.to_csv(self.output_csv_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(self.output_csv_path, mode='w', header=True, index=False)

if __name__=="__main__":
    data = {
    "image_path": "image.jpg",
    "caption": "A group of people on the beach.",
    "questions": ["What are they doing?", "Where are they?"],
    "answers": ["Playing volleyball.", "On the beach."]
}

    writer = CSVWriter("output.csv")
    writer.write(data)
