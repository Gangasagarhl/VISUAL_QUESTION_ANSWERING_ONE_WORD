from image_captioner import ImageCaptioner
from question_generator import QuestionGenerator
from answer_generator import AnswerGenerator

class VQAPipeline:
    """End-to-end VQA pipeline: caption -> question generation -> answer extraction."""

    def __init__(self):
        self.captioner = ImageCaptioner()
        self.qgen = QuestionGenerator()
        self.ansgen = AnswerGenerator()
        print("All the initialisations are done\n")
    def run(self, image_path: str, num_questions: int = 5) -> dict:
        """Run full pipeline on an image."""
        caption = self.captioner.caption(image_path)
        print("Caption: ", caption,"\n")
        questions = self.qgen.generate_questions(caption, num_questions)
        print("Questions: \n",questions)
        answers = [self.ansgen.answer(caption, q) for q in questions]
        print("\n\n", answers,"\n")

        return {
            "image_path": image_path,
            "caption": caption,
            "questions": questions,
            "answers": answers
        }
            
if __name__ ==  "__main__": 
    return_data = VQAPipeline().run(image_path="image.jpg",num_questions=8)
    #print(return_data)