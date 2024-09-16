import torch
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, AdamW
from datasets import load_metric
from sklearn.model_selection import train_test_split

# Function to run on GPU
def train_on_gpu(train_data, val_data):
    
    tokenizer = AutoTokenizer.from_pretrained("Kaludi/chatgpt-gpt4-prompts-bart-large-cnn-samsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("Kaludi/chatgpt-gpt4-prompts-bart-large-cnn-samsum")
    
    # Ensure model is on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        learning_rate=2e-5,
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    
    trainer.train()
    trainer.save_model("./final_model")

# Function to run on CPU
def process_and_check_prompts():
    
    grammar_corrector = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1", device=-1)
    translator = pipeline("translation", model="google-t5/t5-small", device=-1)
    injection_model = pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2", device=-1)

    user_input = input("Please enter a description for the prompt: ")
    corrected = grammar_corrector(user_input)[0]['generated_text']
    translated = translator(corrected, src_lang="auto", tgt_lang="en")[0]['translation_text']
    prompt = f"Create a guide for: {translated}"
    
    
    result = injection_model(prompt)
    if result[0]['label'] == 'injection':
        return True, "Warning: Potential injection detected!"
    return False, "No injection detected."


data_path = "/content/prompts.csv"
data = pd.read_csv(data_path)
data.drop_duplicates(subset='act', inplace=True)
train_data, val_data = train_test_split(data, test_size=0.1)

# Use ThreadPoolExecutor to run tasks in parallel
with ThreadPoolExecutor() as executor:
    future_gpu = executor.submit(train_on_gpu, train_data, val_data)
    future_cpu = executor.submit(process_and_check_prompts)

    
    gpu_result = future_gpu.result()
    cpu_result, message = future_cpu.result()

    print(f"GPU Training Completed: {gpu_result}")
    print(f"Prompt Processing and Injection Check: {message}")
