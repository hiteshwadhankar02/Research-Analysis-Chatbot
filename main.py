import os
import streamlit as st
import pickle
import time
import langchain
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_perplexity import ChatPerplexity
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# loading all environment variables from .env file
load_dotenv()
PERPLEXITYAI_SECREATE_KEY=os.getenv("PERPLEXITYAI_SECREATE_KEY")
os.environ["PERPLEXITYAI_API_KEY"] = PERPLEXITYAI_SECREATE_KEY

#Initialize LLM 
llm = ChatPerplexity(
    model="sonar",
    temperature=0.7,
    pplx_api_key=os.environ["PERPLEXITYAI_API_KEY"],
    max_tokens=500
)

st.title("News Article QA System with PerplexityAI and Streamlit")
st.write("Ask questions about the loaded news articles 🤔❓")

st.sidebar.title("News Article URLs")
urls = []
url_limit = 3
for i in range(url_limit):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "project_vector_store.pxl"

# loading
main_loading_placeholder = st.empty()
if process_url_clicked:
    # load
    loader = UnstructuredURLLoader(urls=urls)
    main_loading_placeholder.text("Loading data from URLs...🔃")
    data = loader.load()
    # split
    main_loading_placeholder.text("Spliting Texts...🪓")
    rec_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " "],
        chunk_size = 1000
    )
    docs = rec_splitter.split_documents(data)
    # embeddingg and save to faiss index
    main_loading_placeholder.text("Starting Embedding...⚡")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    project_vector_store = FAISS.from_documents(docs, embeddings)
    # storing
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_index = pickle.load(f)
        print("Vector index loaded from file.")
    else:
        vector_index = project_vector_store
        with open(file_path, "wb") as f:
            pickle.dump(vector_index, f)
        print("Vector index created and saved to file.")
    
    main_loading_placeholder.text("Process Completed! ✅")

query = main_loading_placeholder.text_input("Enter your question about the articles:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            project_vector_store = pickle.load(f)
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=project_vector_store.as_retriever(),
                return_source_documents=True
            )
            result = result = chain.invoke({"query": query})
            st.header("Answer:")
            st.write(result["result"])

            # display sources if available
            if "source_documents" in result:
                st.subheader("Sources:")
                for doc in result["source_documents"]:
                    st.write(f"- {doc.metadata.get('source', 'Unknown source')}")