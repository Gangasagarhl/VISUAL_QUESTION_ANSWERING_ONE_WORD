## VISUAL_QUESTION_ANSWERING_ONE_WORD
### Step 1: Prepare the Data Question one word answer. (run index_process_all_images.py)
#### 1.Takes the image path from csv
#### 2.Pass the image for captioning through BLIP
#### 3. Based on the caption generated, ollamma generates f"{number_of_questions_fixed}", hence generted that many number of questions.
#### 4. For each and every question there will be one word answer from the ollama. 
#### 5. For each every question answers pair will be stored in the output.csv, which has image path
#### 6. This is done for all the images and the output is saved


### Step 2: Fine tune the BLIP VQA using LORA (finetune_blip_vqa.py)
#### This automatically saves the .pt and then that is loaded to finetune the model again


## Summary

This project presents a structured **Visual Question Answering (VQA)** pipeline using the **Amazon-Berkeley Objects (ABO)** dataset. The objective is to answer open-ended questions about images using **single-word responses**, powered by cutting-edge **Vision-Language Models (VLMs)**.

###  Dataset & Pipeline

- Curated image-description pairs from the ABO dataset
- Extracted English-language descriptions from multilingual JSON metadata
- Encoded image and prompt into a CSV with **Base64-encoded images**
- Each image was queried to generate **10 diverse visual-only Q&A pairs**
- Topics included: Object, Color, Count, Action, Shape, Scene, etc.

###  Models Used

- **BLIP-Base VQA**
- **Qwen2-VL-2B-Instruct**

Both models were benchmarked and later fine-tuned using **LoRA (Low-Rank Adaptation)**, a parameter-efficient fine-tuning technique that significantly reduces training cost by updating fewer parameters (<0.1%).

##  Performance Gains After Fine-Tuning

| Metric              | BLIP-Base ↑ | Qwen2-VL ↑ |
|---------------------|-------------|------------|
| Exact Match (Short) | +26.14%     | +28.60%    |
| BAAI Similarity     | +0.08       | +0.10      |
| BERTScore (F1)      | +0.02       | +0.04      |

 **Up to 28.6% improvement** in Exact Match, alongside notable gains in semantic alignment.

## 🛠 Applications

- Product Cataloging
- Visual Assistive Systems
- Multimodal Information Retrieval
- Natural Language-driven Image Understanding

---

 All code, data preprocessing scripts, and fine-tuning configuration are included in this repository.



