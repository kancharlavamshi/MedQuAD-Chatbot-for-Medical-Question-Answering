# **MedQuAD Chatbot for Medical Question Answering**

![streamlit_app](https://github.com/user-attachments/assets/35d26d46-089d-4662-bf48-07a0f0340ef2)

## **Project Overview**
The **MedQuAD Chatbot** project is a sophisticated **AI-driven conversational agent** designed to provide **accurate and relevant answers** to medical-related questions. This chatbot is trained using the **MedQuAD (Medical Question Answering Dataset)**, a comprehensive collection of real-world question-answer pairs sourced from reliable **National Institutes of Health (NIH)** websites. The goal of this project is to build a model capable of assisting users with health-related queries, ensuring they receive information that is both accurate and easy to understand.

This project uses advanced **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques, leveraging cutting-edge **transformer models** such as **DialoGPT** to generate responses. It aims to help researchers and developers create more effective medical chatbots and intelligent health search engines.

---

## **Key Features**

### 1. **Data Preprocessing**
- The raw **MedQuAD dataset** is cleaned by removing duplicates and irrelevant entries.
- The dataset is processed into a **training-ready format**, with questions and answers stored as **input-output pairs**.
- The final dataset is saved as a **JSON file**, ensuring smooth integration with the training pipeline.

### 2. **Tokenizer and Model**
- The chatbot uses the **DialoGPT** model, a variant of **GPT** (Generative Pre-trained Transformer) fine-tuned for conversational contexts.
- A custom tokenizer is used, adapted to work with special tokens, ensuring proper handling of padding and end-of-sequence tokens during training.

### 3. **Model Training**
- The dataset is split into **training** and **validation sets**, and the model is fine-tuned using the **Hugging Face Transformers** library.
- Key hyperparameters (learning rate, batch size, number of epochs) are **configurable** via command-line arguments for experimentation.
- The model is trained using **Causal Language Modeling**, enabling the model to generate conversational responses to user inputs.

### 4. **Interactive Chatbot**
- After training, the model is deployed as an interactive chatbot.
- The chatbot answers medical queries and provides responses with a special focus on **accuracy** and **clarity**.
- It ensures that the chatbot **avoids repeating user inputs** in its responses, providing a smoother conversational experience.

### 5. **Deployment**
- A **Streamlit-based deployment interface** is provided for users to interact with the trained model through a web-based GUI.
- The interface is responsive and visually appealing, featuring:
  - Gradient backgrounds
  - Message bubbles for both user and bot

---

## **How to Run the Code**

Follow these steps to run the MedQuAD Chatbot locally or in your development environment:

###  **Clone the Repository**
   To get started, first clone the repository to your local machine:
   
   ```bash
   git clone https://github.com/kancharlavamshi/MedQuAD-Chatbot-for-Medical-Question-Answering.git
   cd MedQuAD-Chatbot-for-Medical-Question-Answering
   pip install -r requirements.txt  # install
   ```
## How to Run the Code

### 1. **Preprocessing Data**
<details>
<summary>Preprocess</summary>

The **preprocess.py** script processes the raw MedQuAD dataset by removing duplicates and irrelevant entries. It saves the preprocessed dataset in a JSON file format, ready to be used for training.

To run the preprocessing script, use the following command:
```b
python preprocess.py --input_data /path/to/medquad.csv --output_data processed_data.json
```
- `--input_data`: Path to the raw **MedQuAD** CSV dataset.
- `--output_data`: Path to save the preprocessed dataset in **JSON** format.

This script removes duplicates and prepares the dataset for model training.
</details>

### 2. **Train the Model**
<details>
<summary>Train</summary>

The **train_model.py** script fine-tunes the **DialoGPT** model on the processed MedQuAD dataset. The training script is configurable through command-line arguments, allowing you to modify hyperparameters such as epochs, batch size, and learning rate.

To start training, run the following command:
```
python train_model.py --epochs 30 --batch_size 8 --learning_rate 5e-5 --save_model True --save_path ./models/medquad_model --validation_size 0.1
```
- `--epochs`: Number of training epochs.
- `--batch_size`: The batch size used for training.
- `--learning_rate`: Learning rate for the optimizer.
- `--save_model`: Option to save the trained model after training.
- `--save_path`: Path where the model will be saved.
- `--validation_size`: Fraction of data used for validation during training.

This will start the training process using **Causal Language Modeling** and save the fine-tuned model for later use.
</details>

### 3. **Inference**
<details>
<summary>Inference</summary>

Once the model is trained, the **inference.py** script allows you to interact with the trained model and generate responses to user inputs. The script loads the model and tokenizer, processes the user input, and returns a response based on the trained model.

Run the following command to interact with the trained model:
```
python chatbot_inference.py --model_path /path/to/medquad_model --user_input "What are the symptoms of Glaucoma?"
```
- `--model_path`: Path to the trained **DialoGPT** model.
- `--user_input`: The medical question you want the chatbot to respond to.

This script will generate a response based on the trained model's understanding of the input.
</details>

### 4. **Start the Chatbot (Streamlit Interface)**
<details>
<summary>Chatbot Interface</summary>
Once the model is trained, you can interact with the chatbot through a web-based interface. The chatbot is deployed using **Streamlit**, providing a simple and intuitive user interface to ask medical questions.
To start the chatbot:
  
```
streamlit run chatbot_streamlit.py
```

Once you run the command, the chatbot interface will be available in your web browser. You can ask various medical-related questions, and the chatbot will provide answers based on its training.
</details>

## **File Descriptions**

- **preprocessing.py**: This script processes the raw MedQuAD dataset and prepares it for model training.
- **train_model.py**: This script fine-tunes the **DialoGPT** model using the processed MedQuAD dataset.
- **inference.py**: This script allows users to generate responses from the trained model.
- **app.py**: Implements the chatbot interface using **Streamlit** for web deployment.
- **requirements.txt**: Lists all necessary dependencies for the project (e.g., `transformers`, `streamlit`).
- **medquad.csv**: The original raw **MedQuAD** dataset.
- **processed_data.json**: The preprocessed version of the **MedQuAD** dataset.


### [YouTube](https://youtu.be/LS6z550nfL4?feature=shared)



### About Me
Hello, I’m Vamshi! I’m passionate about technology, machine learning, and innovation. I enjoy solving complex problems through programming and data-driven solutions.

For custom projects or freelance work, feel free to reach out to me on [Upwork](https://www.upwork.com](https://www.upwork.com/freelancers/vamshikrishnak?mp_source=share) or [Fiverr](https://www.fiverr.com/vamshikrishn486?source=post_page-----a5674be25df2). Let’s work together to bring your ideas to life!



