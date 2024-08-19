import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
import langchain
import os
os.environ["GOOGLE_API_KEY"] = 'AIzaSyAPBlOBai9Qc6YSp1Adn-pztn0aanMV34w'


st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

main_placeholder = st.empty()
urls = []
for i in range(3):
    
    url = st.sidebar.text_input(f"URL {i+1}")
    if len(url)>0:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    st.write(data)
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200

    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # vectorindex_openai = FAISS.from_documents(docs, embeddings)
    
