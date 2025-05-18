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


---

 All code, data preprocessing scripts, and fine-tuning configuration are included in this repository.
 Report has been uploaded here [LINK](https://github.com/Gangasagarhl/VISUAL_QUESTION_ANSWERING_ONE_WORD/blob/main/VR_REPORT.pdf).




