import os
import streamlit as st
import pickle
import time
import langchain
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

# WEBPAGE LOADER
loader = UnstructuredURLLoader(urls=[
    "https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html"
])

data = loader.load()
print(f"Loaded {len(data)} documents")

# TEXT SPLITTER
text_splitter = RecursiveCharacterTextSplitter(
    # separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(data)
print(f"Created {len(docs)} chunks")

# OPENAI Embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vextor_index_openAI = FAISS.from_documents(docs, embeddings)

# Stoing the vectore index in local file
file_path = "vector_index_openAI.pkl"
# with open(file_path, "wb") as f:
#     pickle.dump(vextor_index_openAI, f)

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vextor_index = pickle.load(f)
    print("Vector index loaded from file.")
else:
    vextor_index = vextor_index_openAI
    with open(file_path, "wb") as f:
        pickle.dump(vextor_index, f)
    print("Vector index created and saved to file.")

# Create the QA chain
chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = vextor_index.as_retriever())

query = "What is the current market trend according to the articles?"

result = chain({"question": query}, return_only_outputs=True)
print("Answer:", result['answer'])
print("Sources:", result.get("sources", "No sources found"))