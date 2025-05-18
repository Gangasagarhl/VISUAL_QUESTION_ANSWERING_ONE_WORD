import torch
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, logging
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# Suppress warnings
logging.set_verbosity_error()

# Helper to create prompt with image
def create_message(image_path, question):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.csv_path)

    # Load processor & model
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        trust_remote_code=True,
        use_fast=True
    )
    
    # Load the base model only - LoRA loading was failing
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    generated_answers = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            image_path = f"{args.image_dir}/{row['image_name']}"
            question = str(row['question'])

            prompt = f"{question}. Answer the question in **one word** only."
            messages = create_message(image_path, prompt)

            inputs_short = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            
            inputs = processor(
                text=[inputs_short],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                generation_config = {
                    "max_new_tokens": 10,
                    "do_sample": False,
                    "pad_token_id": processor.tokenizer.pad_token_id,
                    "eos_token_id": processor.tokenizer.eos_token_id
                }
                
                if args.debug:
                    print(f"Generation config: {generation_config}")
                
                generated_ids = model.generate(**inputs, **generation_config)
                
                if args.debug:
                    print(f"Input IDs shape: {inputs.input_ids.shape}")
                    print(f"Generated IDs shape: {generated_ids.shape}")
                
                # Extract only the newly generated tokens
                generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
                
                # Decode the generated tokens
                answer = processor.tokenizer.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                if args.debug:
                    print(f"Raw answer: '{answer}'")

            # Clean answer to ensure one word (if there's content)
            if answer and answer.strip():
                answer = str(answer).split()[0].lower()
            else:
                answer = "none"  # Default for empty responses

            if args.debug:
                print(f"Final answer: '{answer}'")

        except Exception as e:
            print(f"[ERROR] at row {idx}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            answer = "error"

        generated_answers.append(answer)

    # Add to DataFrame and save
    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
