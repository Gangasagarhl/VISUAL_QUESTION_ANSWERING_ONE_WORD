import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    """Caption images using BLIP (Base)."""

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        # Set use_fast=True to use the faster processor
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.device = self.model.device

    def caption(self, image_path: str, prompt: str = "a photography of", max_new_tokens: int = 50) -> str:
        """Generate a caption for an image using BLIP Base."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Specify the path to your image file
    image_path = "image.jpg"  # Replace with your image file path
    
    # Create an instance of ImageCaptioner
    captioner = ImageCaptioner()

    # Generate the caption for the provided image
    caption = captioner.caption(image_path)

    # Output the generated caption
    print("Generated Caption:", caption)
