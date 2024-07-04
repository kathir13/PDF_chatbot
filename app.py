import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
import tempfile
from dotenv import load_dotenv

# Set environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_KEY"] = os.getenv("HF_KEY")

# Streamlit app layout
st.title("PDF Question-Answering System")
st.write("Upload a PDF and ask questions about its content.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load PDF and split into documents
    pdf = PyPDFLoader(temp_file_path)
    load = pdf.load()
    document = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc = document.split_documents(load)

    # Initialize embeddings and vectorstore
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.environ.get("HF_KEY"), model_name="sentence-transformers/all-MiniLM-l6-v2")
    db = Chroma.from_documents(doc, embeddings)

    # Initialize language model and prompt template
    llm = ChatGroq(model="llama3-8b-8192", temperature=1)
    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant designed to help with question-answering tasks, especially for last-minute exam preparation.
    Use the following pieces of retrieved context to answer the question as comprehensively as possible. 
    If you don't know the answer based on the given context, simply state, "I don't know." 
    Your goal is to maximize the amount of accurate information provided in your answer.
    If the user asks a question beyond the content available in the provided PDF, respond with, "The information is not specified in the PDF."

    Question: {input}

    Context: {context}""")

    # Create chains
    chain = create_stuff_documents_chain(llm, prompt=prompt)
    retriever = db.as_retriever()
    ret_chain = create_retrieval_chain(retriever, chain)
    st.write("PDF uploaded and processed successfully.")

    # Question input
    question = st.text_input("Enter your question here:")

    if question:
        with st.spinner('Processing...'):
            try:
                result = ret_chain.invoke({"input": question})["answer"]
                st.write("Answer:")
                st.write(result)
            except Exception as e:
                st.error(f"Error: {e}")
                st.write("Please check the API keys and permissions.")
