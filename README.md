# 🏥 Medical AI Assistant (RAG + OpenAI)

An AI-powered **Medical Question Answering System** built using **Retrieval-Augmented Generation (RAG)** and **Streamlit**.

This project retrieves relevant medical knowledge from a dataset and generates context-aware answers using OpenAI models.

---

## 🚀 Features

- 💬 Chat-based medical assistant
- 📚 Retrieval-Augmented Generation (RAG)
- 🧠 Context-aware answers using embeddings
- ⚡ Fast semantic search (FAISS)
- 🎨 Interactive UI with Streamlit
- 📖 Source-based answers (transparency)
- ⚠️ Built-in medical disclaimer

---

## 🏗️ Architecture
```
User Query
↓
Embedding Model
↓
Vector Search (FAISS)
↓
Relevant Documents
↓
LLM (OpenAI)
↓
Final Answer + Sources
```
## 📂 Project Structure
```
medical-assistant/
│
├── app.py # Streamlit frontend
├── requirements.txt # Dependencies
├── .env # API key
├── medDataset_processed.csv # Dataset
│
└── src/
  ├── config.py # Config settings
  ├── data_loader.py # Load data
  ├── vector_store.py # FAISS + embeddings
  ├── retriever.py # Retrieve docs
  └── rag.py # RAG pipeline
```

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/ranjeet-singh45/medical-assistant.git
cd medical-assistant
python -m venv venv
```
Activate it:

Windows:
```
venv\Scripts\activate
```
Mac/Linux:
```
source venv/bin/activate
```
Install Dependencies
```
pip install -r requirements.txt
```
Create a .env file in root:
```
OPENAI_API_KEY=your_openai_api_key
```
Run the App
```
streamlit run app.py
