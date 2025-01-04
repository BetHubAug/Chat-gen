import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Initialize Chatbot Model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def generate_chatbot_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

# Initialize Image Generation Model
image_generator = pipeline("image-generation", model="CompVis/stable-diffusion-v1-4")

def generate_image(prompt):
    images = image_generator(prompt)
    return images[0]

def chatbot_interface(user_input):
    chat_response = generate_chatbot_response(user_input)
    image = generate_image(user_input)
    return chat_response, image

iface = gr.Interface(fn=chatbot_interface, inputs="text", outputs=["text", "image"], title="Chat Bot with Image Generation")
iface.launch()
