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
