# Load the file as a document 
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("Resume -Surya.pdf")
documents = loader.load()

# Make the document into chunks 
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# Convert the doc into embeddings using OpenAIEmbeddings 
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Store the embeddings into a FAISS vector store for retrieval 
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# RAG Pipeline 
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

retriever = vectorstore.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)


# test query 
query = "What are Surya's skills ?"
response = qa.run(query)

print(response)


# UI 
# import streamlit as st

# st.title("My Resume Chatbot")
# query = st.text_input("Ask something about me:")
# if query:
#     response = qa.run(query)
#     st.write(response)
