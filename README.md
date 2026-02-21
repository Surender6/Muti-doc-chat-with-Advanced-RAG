
# 🧠 Multi-Document Chat with Advanced RAG & Indexing

An end-to-end conversational AI that lets users upload and query **multiple documents** using advanced **Retrieval-Augmented Generation (RAG)** with indexing and semantic search.

This system extracts text, embeds chunks, builds a vector index, retrieves relevant chunks, and generates contextual responses using a Large Language Model.

---

## 🚀 Project Overview

Millions of documents contain useful insights — but searching across them manually is slow and inefficient.

This project solves that problem by letting users ask questions in natural language and get **context-aware answers**, grounded in multiple uploaded documents.

It demonstrates:

✔ Document ingestion and indexing  
✔ Semantic search with vector similarity  
✔ Context-aware response generation (RAG)  
✔ A modular Python codebase that can be extended easily

---

## 🏗 Architecture

```plaintext
User Query
    ↓
Document Upload / Text Extraction
    ↓
Chunking + Embedding Generation
    ↓
Indexing in Vector Store
    ↓
Similarity Search (Top-k)
    ↓
Context + LLM → Final Response

## 🛠️ Tech Stack

- Python
- LangChain
- FAISS 
- HuggingFace LLM
- Streamlit (if UI exists)
- AWS EC2 

## 🔍 Key Features

- Multi-document ingestion
- Semantic chunking
- Vector similarity search
- Context-aware response generation
- Unified indexing
- Conversational memory
- Modular architecture

## 📊 Retrieval Strategy

- Chunk size: 200 tokens
- Overlap: 20
- Embedding model: XXXX
- Similarity metric: cosine similarity
- Top-k retrieval: k=4
- Optional: Re-ranking (if implemented)

## 📈 Sample Interaction

User: What are the key findings in the annual report?
System: ...

## ⚙️ Setup Instructions

### 1. Clone the repository
git clone <your_repo_url>

### 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add your API key
Create a .env file:
OPENAI_API_KEY=your_key_here

### 5. Run the application
streamlit run app.py


## 🌐 Deployment Guide

This project can be deployed using AWS EC2 (production-ready).


---

### 🚀 Deploy on AWS EC2 (Production Setup)

#### Step 1: Launch EC2
- Ubuntu 22.04
- Open ports: 22 (SSH) and 8501 (Streamlit)

#### Step 2: Connect to EC2

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip

#### step 3: Install Dependencies
sudo apt update
sudo apt install python3-pip python3-venv -y

#### Step 4: Clone Repository
git clone https://github.com/Surender6/Muti-doc-chat-with-Advanced-RAG.git
cd Muti-doc-chat-with-Advanced-RAG

#### Step 5: Setup Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

#### step 6: Set Environment Variables
export OPENAI_API_KEY="your_api_key"

#### Step 7: Run App
streamlit run main.py --server.port 8501 --server.address 0.0.0.0

Now access:

http://your-ec2-ip:8501

