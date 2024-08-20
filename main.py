import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQAWithSourcesChain
import os
import nltk
nltk.download('all')
os.environ["GOOGLE_API_KEY"] = 'AIzaSyAPBlOBai9Qc6YSp1Adn-pztn0aanMV34w'

st.title("Gen AI: News Research Tool 📈")
initial_sidebar_state="expanded"
# Initialize session state
if "URLS_INPUT" not in st.session_state:
    st.session_state.URLS_INPUT = []
if "check" not in st.session_state:
    st.session_state.check = False
if "vectorindex_openai" not in st.session_state:
    st.session_state.vectorindex_openai = None

main_placeholder = st.empty()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

st.sidebar.title("Enter Your Url's Here")
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i+1}")
    if len(url) > 0:
        st.session_state.URLS_INPUT.append(url)

process_url_clicked = st.sidebar.button("Process Url's")

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=st.session_state.URLS_INPUT)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.vectorindex_openai = FAISS.from_documents(docs, embeddings)
    st.session_state.vectorindex_openai.save_local("faiss_index")
    st.write(st.session_state.vectorindex_openai)
    st.session_state.check = True
    main_placeholder.text("Process Complete Now You can Ask Question...✅✅✅")

if st.session_state.check:
    query = st.text_input('You Can Ask Your Question Now')
    if query:
        vectorindex_openai = FAISS.load_local(
            "faiss_index", 
            GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
            allow_dangerous_deserialization=True
        )
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex_openai.as_retriever())
        response = chain({"question": query}, return_only_outputs=True)
        st.write("Answer --- ",response['answer'])
        st.write("Source --- ",response['sources'])


