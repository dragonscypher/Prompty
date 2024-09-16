import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM,TFAutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor

model = TFAutoModelForSeq2SeqLM.from_pretrained("Kaludi/chatgpt-gpt4-prompts-bart-large-cnn-samsum")
tokenizer = AutoTokenizer.from_pretrained("Kaludi/chatgpt-gpt4-prompts-bart-large-cnn-samsum")


grammar_corrector = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1", device=-1)
translator = pipeline("translation", model="google-t5/t5-small", device=-1)
injection_model = pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2", device=-1)

def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=5.0, num_beams=2)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_text(text):
    corrected = grammar_corrector(text)[0]['generated_text']
    translated = translator(corrected, src_lang="auto", tgt_lang="en")[0]['translation_text']
    return translated

def check_prompt(prompt):
    result = injection_model(prompt)
    if result[0]['label'] == 'injection':
        return "Warning: Potential injection detected!"
    return "No injection detected."

def generate_and_check_prompt():
    user_input = input("Please enter a description for the prompt: ")
    processed_input = process_text(user_input)
    summarized_input = summarize_text(processed_input)
    prompt = f"Create a guide for: {summarized_input}"
    injection_message = check_prompt(prompt)
    
    return prompt, injection_message

if __name__ == "__main__":
    prompt, message = generate_and_check_prompt()
    print(f"Generated Prompt: {prompt}")
    print(message)
