# inference/chatbot_inference.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("/path/to/medquad_tuned_model")
model = AutoModelForCausalLM.from_pretrained("/path/to/medquad_tuned_model")

# Add special tokens
tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|bot|>"]})
model.resize_token_embeddings(len(tokenizer))

# Chat function
def chat_with_bot(input_text, model, tokenizer):
    formatted_input = f"<|user|> {input_text.strip()} <|bot|>"
    input_ids = tokenizer.encode(formatted_input, return_tensors="pt")

    # Generate response
    response_ids = model.generate(input_ids, max_length=512, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True, top_p=0.9, top_k=50, repetition_penalty=1.5)

    # Decode and extract bot's response
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    response = response.split("<|bot|>")[-1].strip()  # Remove input repetition

    return response

# Example interaction
print("Chatbot is ready! Type 'exit' to stop the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    bot_response = chat_with_bot(user_input, model, tokenizer)
    print(f"Chatbot: {bot_response}")
