 📊 News Research Analysis Chatbot
 
An AI-powered chatbot for **News Research Analysis** that fetches data from news articles, processes them into embeddings, and allows users to ask **contextual questions** about the articles.

This project automates the tedious process of copying & pasting large articles into ChatGPT and instead provides a *streamlined research assistant* powered by *LangChain, HuggingFace, Perplexity API, and FAISS*.

🚀 Features

* 🔎 *Automated Web Scraping*: Fetches articles from multiple news sources.
* 📑 *Chunking & Splitting*: Breaks down large articles into smaller chunks for efficient search.
* 🧠 *Vector Search with FAISS*: Stores embeddings for fast and scalable retrieval.
* 🤖 *Perplexity-Powered Chatbot*: Answers questions contextually based on retrieved knowledge.
* ⚡ *Streamlit UI*: Easy-to-use interface for interacting with the chatbot.

🛠️ Tech Stack

* *Frameworks & Tools*:

  * [LangChain](https://www.langchain.com/)
  * [Streamlit](https://streamlit.io/)

* *LLM & Embeddings*:

  * [Perplexity API](https://www.perplexity.ai/) (LLM for answering queries)
  * [HuggingFace Embeddings](https://huggingface.co/)

* Vector Database*:

  * [FAISS](https://faiss.ai/) (for similarity search & retrieval)

🔄 Workflow

flowchart LR
    A[News Articles] --> B[Web Scraper]
    B --> C[Text Splitting & Chunking]
    C --> D[HuggingFace Embeddings]
    D --> E[FAISS Vector DB]
    E --> F[Retriever]
    F --> G[Perplexity LLM]
    G --> H[Answer to User]

⚙️ Architecture

1. *Data Ingestion*

   * Fetch articles using UnstructuredURLLoader
   * Split documents into chunks (RecursiveCharacterTextSplitter)

2. *Embedding & Storage*

   * Convert chunks into embeddings (HuggingFaceEmbeddings)
   * Store vectors in *FAISS*

3. *Query Handling*

   * Retrieve relevant chunks using similarity search
   * Pass context to *Perplexity LLM*
   * Generate final answer

📦 Installation

1. Clone the repo:

   bash
   git clone https://github.com/hiteshwadhankar02/Research-Analysis-Chatbot.git
   cd Research-Analysis-Chatbot
   
2. Create a virtual environment & install dependencies:

   bash
   pip install -r requirements.txt
   

3. Setup environment variables (.env):

   env
   PERPLEXITY_API_KEY=your_api_key_here
   
▶️ Usage

Run the chatbot with Streamlit:

bash
streamlit run app.py


Upload article URLs or provide links → Ask questions → Get contextual answers!

📚 Example Use Case

1. Fetch articles about a stock (e.g., *Infosys quarterly results*)
2. Store them in FAISS after embedding
3. Ask:

   > "What were the key highlights from Infosys Q1 earnings?"
4. Get a concise, research-backed answer 🎯

🧩 Classes & Modules Used

* *Text Loaders*: UnstructuredURLLoader
* *Splitters*: RecursiveCharacterTextSplitter
* *Embeddings*: HuggingFaceEmbeddings
* *Vector DB*: FAISS
* *Retriever*: RetrievalQA

🔮 Future Enhancements

* 🌐 Multi-source scraping (Yahoo Finance, MoneyControl, etc.)
* 📊 Dashboard for financial insights
* 🧾 PDF/CSV support for analyst reports
* 🕵️ Sentiment analysis on market news
