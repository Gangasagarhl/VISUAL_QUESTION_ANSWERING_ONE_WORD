import ollama

class AnswerGenerator:
    """Generate one-word answer to a question based on an image caption using Ollama llama3.2."""

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def answer(self, caption: str, question: str, max_new_tokens: int = 10) -> str:
        """Answer the question based on the caption in one word using Ollama."""
        prompt = (
            f"Image description: \"{caption}\"\n"
            f"Question: \"{question}\"\n"
            "Answer in one word:"
        )

        try:
            response = ollama.chat(model=self.model_name, messages=[
                {"role": "user", "content": prompt}
            ])
            answer = response['message']['content'].strip()
            # Ensure one word
            return answer.split()[0] if answer else "No answer"
        except Exception as e:
            print("Error calling Ollama:", e)
            return "Error"

if __name__ == "__main__":
    ag = AnswerGenerator()
    caption = "A man is holding an umbrella in the rain."
    question = "What is he holding?"
    print("Answer:", ag.answer(caption, question))
