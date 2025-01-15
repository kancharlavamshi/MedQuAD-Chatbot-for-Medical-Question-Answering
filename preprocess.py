# preprocessing/preprocess.py

import pandas as pd
from datasets import Dataset
import json

def preprocess_data(input_csv, output_json):
    """
    Preprocesses the MedQuAD dataset into a JSON format for training.
    Args:
        input_csv (str): Path to the raw dataset (CSV).
        output_json (str): Path to save the processed data (JSON).
    """
    # Load the dataset
    data = pd.read_csv(input_csv)
    
    # Remove duplicate questions
    data = data.drop_duplicates(subset=["question", "answer"])

    # Convert Pandas DataFrame to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(data)
    hf_dataset = hf_dataset.rename_column("question", "input_text")
    hf_dataset = hf_dataset.rename_column("answer", "label")

    # Select relevant columns
    hf_dataset = hf_dataset.select_columns(["input_text", "label"])

    # Filter out invalid rows
    hf_dataset = hf_dataset.filter(lambda x: x["input_text"] is not None and x["label"] is not None)

    # Save as JSON
    conversation_pairs = [{"input": input_text, "response": label} for input_text, label in zip(hf_dataset['input_text'], hf_dataset['label'])]
    
    with open(output_json, 'w') as json_file:
        json.dump(conversation_pairs, json_file, indent=4)

    print(f"Preprocessing complete. Data saved to {output_json}")

# Run this script
if __name__ == "__main__":
    preprocess_data('/data/vamshi/nlp/medquad.csv', './medquad_data.json')
