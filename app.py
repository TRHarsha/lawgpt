import streamlit as st
import os
import pandas as pd
import tensorflow as tf  # TensorFlow for .h5 model support
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Set Streamlit page configuration
st.set_page_config(page_title="LawGPT", layout="wide")

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Load dataset and validate content
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_json(file_path)
        if 'JUDGMENT' not in data.columns:
            raise ValueError("'JUDGMENT' column missing.")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data_path = "2.json"
data = load_data(data_path)

if data is None:
    st.stop()  # Stop if data loading fails

st.write("Data Sample:", data.head())

# Initialize embeddings and vector store
try:
    st.write("Initializing embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_texts(data['JUDGMENT'].tolist(), embedding=embeddings, persist_directory="my_vector_store")
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Retrieve top 5 similar docs
    st.success("Vector store initialized successfully.")
except Exception as e:
    st.error(f"Error initializing vector store: {e}")
    st.stop()

# Load the trained legal model from .h5 file
try:
    legal_model = tf.keras.models.load_model("legal_info_model.h5")
except (OSError, ImportError) as e:
    st.warning(f"Model could not be loaded: {e}")
    legal_model = None

# Initialize session state for chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Define prompt template for the LLM
prompt_template = """
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

# Initialize the LLM (ChatGroq with Llama)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# Create Conversational Retrieval Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)

st.title("LawGPT: Legal Assistant ChatBot")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function to generate responses
def generate_response(input_prompt):
    """Generate a response using either the legal model or the LLM."""
    if legal_model and "legal" in input_prompt.lower():  # Use custom legal model if relevant
        st.write("Using the legal model...")
        prediction = legal_model.predict([input_prompt])
        return prediction[0]
    else:
        st.write("Using the LLM with document retrieval...")
        result = qa.invoke(input=input_prompt)
        return result.get("answer", "Sorry, I couldn't find a relevant answer.")

# Handle user input
input_prompt = st.chat_input("Ask a legal question...")

if input_prompt:
    # Display user's question
    st.chat_message("user").write(input_prompt)
    st.session_state.messages.append({"role": "user", "content": input_prompt})

    # Generate response
    response = generate_response(input_prompt)

    # Check if the retrieved answer is relevant
    if "no information available" in response.lower():
        st.warning("No relevant information found. Please refine your query.")
        with open("unanswered_questions.txt", "a") as f:
            f.write(f"{input_prompt}\n")
    else:
        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Button to reset the conversation
if st.button("Reset All Chat 🗑️"):
    st.session_state.messages.clear()
    st.session_state.memory.clear()
    st.success("Chat history reset.")

# Button to persist vector store
if st.button("Persist Vector Store"):
    try:
        db.persist()
        st.success("Vector store persisted successfully.")
    except Exception as e:
        st.error(f"Error persisting vector store: {e}")
