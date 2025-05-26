import torch
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
import logging
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_message(image_path, question):
    """Create a message dictionary for the model."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }
    ]

def validate_inputs(image_dir, csv_path):
    """Validate input directory and CSV file."""
    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory '{image_dir}' does not exist.")
    if not os.path.isfile(csv_path):
        raise ValueError(f"CSV file '{csv_path}' does not exist.")
    df = pd.read_csv(csv_path)
    required_columns = ['image_name', 'question']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Run inference on images with questions using Qwen2-VL model.")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Validate inputs
    try:
        df = validate_inputs(args.image_dir, args.csv_path)
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        return

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load processor & model
    try:
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
            use_fast=True
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Base model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model or processor: {e}")
        return

    # Freeze vision encoder and embeddings
    for name, param in model.named_parameters():
        if "vision_tower" in name or "embed_tokens" in name:
            param.requires_grad = False

    # Load LoRA weights if available
    try:
        model = PeftModel.from_pretrained(model, "mbashish/qwen_finetuned", is_trainable=False)
        logger.info("LoRA weights loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load LoRA weights: {e}. Continuing with base model.")

    model.eval()
    model.to(device)
    generated_answers = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        try:
            image_path = os.path.join(args.image_dir, row['image_name'])
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Validate image
            try:
                Image.open(image_path).verify()
            except Exception as e:
                raise ValueError(f"Invalid image file {image_path}: {e}")

            question = str(row['question']).strip()
            if not question:
                raise ValueError("Question is empty.")

            prompt = f"{question}. Answer the question in **one word** only."
            messages = create_message(image_path, prompt)

            # Prepare inputs
            inputs_short = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[inputs_short],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            # Generate answer
            with torch.no_grad():
                generation_config = {
                    "max_new_tokens": 5,  # Reduced for one-word answers
                    "do_sample": False,
                    "pad_token_id": processor.tokenizer.pad_token_id,
                    "eos_token_id": processor.tokenizer.eos_token_id
                }
                
                logger.debug(f"Generation config: {generation_config}")
                
                generated_ids = model.generate(**inputs, **generation_config)
                
                logger.debug(f"Input IDs shape: {inputs.input_ids.shape}")
                logger.debug(f"Generated IDs shape: {generated_ids.shape}")
                
                # Extract newly generated tokens
                generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
                
                # Decode
                answer = processor.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0].strip()

                logger.debug(f"Raw answer: '{answer}'")

            # Ensure one-word answer
            if answer:
                answer = answer.split()[0].lower()
            else:
                answer = "none"

            logger.debug(f"Final answer: '{answer}'")

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logger.error(f"Error at row {idx}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            answer = "error"

        generated_answers.append(answer)

        # Clear memory
        if device == "cuda":
            torch.cuda.empty_cache()

    # Save results
    try:
        df["generated_answer"] = generated_answers
        logger.info("\nFirst few rows of results:")
        logger.info(df.head().to_string())
        df.to_csv("results.csv", index=False)
        logger.info("Results saved to 'results.csv'")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    main()
