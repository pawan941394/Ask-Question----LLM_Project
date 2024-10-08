import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
import nltk
import os

# Check if the NLTK data directory exists
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download NLTK data if not already present
nltk.data.path.append(nltk_data_dir)

with st.spinner("Downloading NLTK data..."):
    nltk.download('all', download_dir=nltk_data_dir)
    st.success("NLTK data downloaded successfully!")

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = st.secrets['api_key']
st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center;">
        <a href="https://www.instagram.com/p_awan__kumar/" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" width="40" style="margin-right: 10px;">
        </a>
        <a href="https://www.linkedin.com/in/pawan941394/" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="40" style="margin-right: 10px;">
        </a>
        <a href="https://www.youtube.com/channel/UClgbj0iYh5mqY_81CMCw25Q/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/1384/1384060.png" width="40" style="margin-right: 10px;">
        </a>
        <a href="https://github.com/pawan941394" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="40" style="margin-right: 10px;">
        </a>
   <a href="https://wa.me/919057714590" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" width="40" style="margin-right: 10px;">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
    
st.title("Gen AI: News Research Tool 📈")
st.sidebar.title("Enter Your URLs Here")

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

# Sidebar input for URLs
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i+1}")
    if len(url) > 0:
        st.session_state.URLS_INPUT.append(url)

# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=st.session_state.URLS_INPUT)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.vectorindex_openai = FAISS.from_documents(docs, embeddings)
    st.session_state.vectorindex_openai.save_local("faiss_index")

    st.session_state.check = True
    main_placeholder.success("Process Complete. Now you can ask questions...✅✅✅")

# Query section
if st.session_state.check:
    query = st.text_input('You can ask your question now')
    if query:
        vectorindex_openai = FAISS.load_local(
            "faiss_index", 
            GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
            allow_dangerous_deserialization=True
        )
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex_openai.as_retriever())
        response = chain({"question": query}, return_only_outputs=True)
        st.write("Answer --- ", response['answer'])
        st.write("Source --- ", response['sources'])
