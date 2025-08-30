import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt

# --- 1. Setup ---
# Load API key from your .env file
load_dotenv()

st.header('Research Tool 🚀')

# Initialize the Language Model
try:
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation"
    )
    model = ChatHuggingFace(llm=llm)
    model_is_ready = True
except Exception as e:
    st.error(f"Error initializing model: {e}")
    model_is_ready = False

# --- 2. User Interface and Logic ---
# Get input from the user using dropdowns
paper_input = st.selectbox(
    "Select Research paper name",
    ["Select....", "Attention is all u need", "BERT : Pre-training of deep bidirectional transformers", "GPT-3: Language Modelsare few-shot learners", "Diffusion models beat GANs on image synthesis"]
)
style_input = st.selectbox(
    "Select explanation style",
    ["Beginner-friendly", "technical", "code-oriented", "mathematical"]
)
length_input = st.selectbox(
    "select explanation length",
    ["short (1-2 paragraphs)", "medium (3-5 paragraphs)", "long (detailed explanation)"]
)

# Template for the prompt

template = load_prompt('template.json')
# This is the button that will trigger the model call
if st.button('Summarize'):
    # Check if a valid paper is selected and if the model is ready
    if model_is_ready and paper_input != "Select....":
        with st.spinner('Generating summary... Please wait.'):
            try:
                # Format the prompt with the user's selections from the dropdowns
                prompt = template.invoke({
                    'paper_input': paper_input,
                    'style_input': style_input,
                    'length_input': length_input
                })
                
                # Invoke the model
                result = model.invoke(prompt)
                
                # Display the result in the web app
                st.markdown(result.content)
            
            except Exception as e:
                st.error(f"An error occurred while generating the summary: {e}")
    elif paper_input == "Select....":
        st.warning("Please select a research paper first.")
    elif not model_is_ready:
        st.error("Model is not ready. Please check the model initialization.")