import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

# Path to your fine-tuned model
model_path = "/path/to/medquad_tuned_model"  # Adjust this to your actual model path

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Use the text-generation pipeline instead of conversational
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to remove the user input from the bot's response
def remove_repeated_input(user_input, bot_response):
    """
    Removes repeated parts of the user's input from the bot's response.
    """
    user_input_lower = user_input.lower().strip()
    bot_response_lower = bot_response.lower().strip()

    # Remove user input if it appears at the start of bot's response
    if bot_response_lower.startswith(user_input_lower):
        bot_response = bot_response[len(user_input):].strip()

    # Also remove user input anywhere in the response
    bot_response = bot_response.replace(user_input, "").strip()

    return bot_response

# Preprocess user input to remove unwanted spaces or punctuation
def preprocess_input(user_input):
    """
    Preprocess the user input to remove extra spaces and normalize the phrasing.
    """
    # Remove extra spaces and fix punctuation
    user_input = user_input.strip()
    
    # Remove spaces before a question mark
    if user_input.endswith(" ?"):
        user_input = user_input[:-2] + "?"
    
    # Normalize question phrasing
    user_input = " ".join(user_input.split())

    return user_input

# Custom CSS styling for Streamlit UI
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #8E2DE2, #4A00E0); /* Gradient background */
        color: white;
        font-family: 'Helvetica', sans-serif;
    }

    .title {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
        color: #FFD700;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
    }

    .project {
        font-size: 20px;
        text-align: center;
        margin-top: 10px;
        color: #87CEEB; /* Sky Blue color */
    }

    .chat-box {
        padding: 20px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.1);
        margin-top: 30px;
        max-height: 500px;
        overflow-y: auto;
    }

    .user-message, .bot-message {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 15px;
        max-width: 80%;
    }

    .user-message {
        background-color: #0078D4;
        color: white;
        align-self: flex-end;
    }

    .bot-message {
        background-color: #28A745;
        color: white;
        align-self: flex-start;
    }

    .input-field {
        font-size: 18px;
        padding: 10px;
        width: 100%;
        border: 2px solid #FFD700;
        border-radius: 5px;
        margin-top: 20px;
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }

    .clear-button {
        background-color: #FF4500;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">MedQuAD Chatbot</div>', unsafe_allow_html=True)

# Add Project Done By: Vamshi in sky blue color
st.markdown('<div class="project">Project done by: Vamshi</div>', unsafe_allow_html=True)

st.write("Welcome! Ask me anything about medical conditions and treatments.")

# Initialize conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Check if the user input already exists in session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# User input
user_input = st.text_input("You: ", key="user_input_input", value=st.session_state.user_input)

# Clear button functionality
if st.button("Clear Conversation", key="clear_button", help="Clear the conversation history"):
    st.session_state.conversation_history = []

if user_input:
    if user_input.lower() == "exit":
        st.write("Goodbye!")
        st.stop()  # Stop the Streamlit app if the user types "exit"

    # Preprocess user input to normalize spaces and punctuation
    user_input = preprocess_input(user_input)

    # Add user input to conversation history
    st.session_state.conversation_history.append(f"You: {user_input}")

    # Generate chatbot response
    input_text = f"<|user|> {user_input} <|bot|>"
    response = chatbot(input_text, max_length=512, num_return_sequences=1)

    # Extract the bot's response
    bot_response = response[0]["generated_text"].replace(input_text, "").strip()

    # Remove repeated user input from the bot's response
    bot_response = remove_repeated_input(user_input, bot_response)

    # Add bot's response to the conversation history
    st.session_state.conversation_history.append(f"Bot: {bot_response}")

    # Update the session state with the empty string to reset the input field
    st.session_state.user_input = ""

# Display conversation history in chat window
st.markdown('<div class="chat-box">', unsafe_allow_html=True)
for message in st.session_state.conversation_history:
    if "You" in message:
        st.markdown(f'<div class="user-message">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{message}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st

# Your chatbot and other app code here...

# Custom CSS to position the disclaimer at the bottom
st.markdown("""
    <style>
        .disclaimer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px;
            text-align: center;
            font-size: 14px;
        }
    </style>
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This chatbot is designed for educational purposes only. The information provided by the chatbot is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    </div>
""", unsafe_allow_html=True)
