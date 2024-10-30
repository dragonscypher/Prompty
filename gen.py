import os
import warnings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress all user warnings

# Load a different model to avoid issues with the previous one
model_name = "facebook/bart-large-cnn"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load models for grammar correction, translation, and prompt injection check (all on CPU)
grammar_corrector = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1", device=-1)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en", device=-1)
injection_model = pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2", device=-1)

# Function to summarize text
def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to process text: grammar correction and translation
def process_text(text):
    corrected = grammar_corrector(text)[0]['generated_text']
    # Translate to English if the input is in another language
    translated = translator(corrected)[0]['translation_text']
    return translated

# Function to check for injection in generated prompt
def check_prompt(prompt):
    result = injection_model(prompt)
    return "Warning: Potential injection detected!" if result[0]['label'] == 'injection' else "No injection detected."

# Main function to handle user input and generate prompt
def generate_and_check_prompt():
    user_input = input("Please enter a clear and concise description for the prompt: ")
    processed_input = process_text(user_input)
    summarized_input = summarize_text(processed_input)
    prompt = f"Create a detailed guide for: {summarized_input}"
    injection_message = check_prompt(prompt)
    
    return prompt, injection_message

# Running the main function
if __name__ == "__main__":
    prompt, message = generate_and_check_prompt()
    print(f"Generated Prompt: {prompt}")
    print(message)
