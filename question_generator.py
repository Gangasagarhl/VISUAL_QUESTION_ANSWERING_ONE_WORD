import ollama

class QuestionGenerator:
    """Generate one-word-answer questions based on an image caption using Ollama."""

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def generate_questions(self, caption: str, num_questions: int = 10) -> list[str]:
        # Few-shot prompt to show desired output format
        prompt = f"""
Here are examples of how I want the output formatted:

Example 1:
Caption: "A cat sits on a windowsill."
Questions:
- What animal is shown?
- Where is the cat sitting?
- What time of day is it?
- Is the animal indoors?
- What color is the cat’s fur?
- Is cat is on Aeroplane?
- What is the color of the cat?
-Is do playing wiht cat?

Example 2:
Caption: "A man is holding an umbrella in the rain."
Questions:
- What is the man holding?
- What is the weather?
- Is he inside?
- What color is the umbrella?
- Is it daytime?

Now you try and you need to generate {num_questions} questions based on the below caption.

Caption: "{caption}"
Questions:
""".strip()

        # Call the Ollama model
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response['message']['content']
        
        # 1) Debug: print raw output to see model's response
        #print("⟶ RAW MODEL OUTPUT:\n", content)

        # 2) Parse out numbered lines
        questions = [
            line.strip().lstrip("0123456789.-").strip()  # Remove any numbering (e.g., '1.', '2.', etc.)
            for line in content.splitlines()
            if line.strip() and line.strip()[0].isdigit()  # Check if the line starts with a number (for numbered questions)
        ]
        
        # Check if no questions were generated
        if not questions:
            print("No questions generated. Check the raw model output above for insights.")

        return questions

if __name__ == "__main__":
    caption = "A woman is holding a red apple in a garden."
    generator = QuestionGenerator()
    questions = generator.generate_questions(caption)

    print("\nGenerated Questions:")
    for q in questions:
        print("-", q)






































"""
class QuestionGenerator:
    '''Generate one-word-answer questions based on an image caption using Ollama.'''

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def generate_questions(self, caption: str, num_questions: int = 5, max_new_tokens: int = 200) -> list[str]:
        prompt = (
            f''' Prompt: [caption:  A woman is holding a red apple in a garden. 
            Questions: What is the gender of person?-What the person is holding?-What is the color of the Apple?-Is she in Aeroplace?-Is she on bike?Where is she now?-where did she found apple?"]
            caption: \"{caption}\",\n"
            generate a list of plausible questions whose answer would be a single word.\n"
            List {num_questions} questions as bullet points.'''
        )

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        questions = response['message']['content'].splitlines()
        questions = [line.strip("- ").strip() for line in questions if line.strip().startswith("-")]
        return questions

        
"""
