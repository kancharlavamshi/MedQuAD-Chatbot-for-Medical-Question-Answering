import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import pandas as pd
import json


# Step 1: Load and preprocess the dataset
def load_data(input_json):
    """
    Load preprocessed data from the JSON file and convert it into a Hugging Face Dataset.
    """
    with open(input_json, 'r') as file:
        conversations = json.load(file)
    
    # Clean the data: remove invalid entries
    cleaned_data = []
    for conv in conversations:
        if conv["input"] and conv["response"]:
            cleaned_data.append({"input": conv["input"], "response": conv["response"]})

    # Create a dataset dictionary
    data = {"text": []}
    for conv in cleaned_data:
        # Concatenate input and response for language modeling
        combined_text = f"<|startoftext|>{conv['input']} {conv['response']}<|endoftext|>"
        data["text"].append(combined_text)

    # Convert to Hugging Face dataset
    dataset = Dataset.from_dict(data)
    return dataset


# Step 2: Tokenize the dataset
def tokenize_data(dataset, tokenizer):
    """
    Tokenizes the dataset using the given tokenizer.
    """
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # Add labels for causal language modeling
    tokenized_dataset = tokenized_dataset.map(lambda examples: {"labels": examples["input_ids"]})
    return tokenized_dataset


# Step 3: Fine-tune the model
def fine_tune_model(tokenized_dataset, tokenizer, output_dir, num_epochs, learning_rate, batch_size):
    """
    Fine-tunes a pretrained model (DialoGPT-small) on the tokenized dataset.
    """
    # Load the model
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings if needed

    # Split the dataset into training and validation sets
    train_dataset = tokenized_dataset.train_test_split(test_size=0.1)['train']
    val_dataset = tokenized_dataset.train_test_split(test_size=0.1)['test']

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,  # Output directory
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        learning_rate=learning_rate,  # Learning rate
        per_device_train_batch_size=batch_size,  # Training batch size
        per_device_eval_batch_size=batch_size,    # Evaluation batch size
        num_train_epochs=num_epochs,             # Number of epochs
        weight_decay=0.01,               # Weight decay
    )

    # Set up the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model training complete. Saved to {output_dir}.")


# Step 4: Main entry point
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fine-tune DialoGPT on MedQuAD dataset")
    parser.add_argument('--input_json', type=str, required=True, help="Path to the preprocessed MedQuAD data (JSON format)")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where the trained model will be saved")
    parser.add_argument('--num_epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate for training")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training and evaluation")

    # Parse arguments
    args = parser.parse_args()

    # Load the preprocessed MedQuAD data (assuming it's saved as JSON)
    dataset = load_data(args.input_json)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    
    # Ensure that a padding token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    # Tokenize the data
    tokenized_dataset = tokenize_data(dataset, tokenizer)
    
    # Fine-tune the model
    fine_tune_model(tokenized_dataset, tokenizer, args.output_dir, args.num_epochs, args.learning_rate, args.batch_size)
