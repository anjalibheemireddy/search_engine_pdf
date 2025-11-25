# 📘 PDF Search Engine 

This project is a **Streamlit-based PDF Search Engine** that uses a  **RAG (Retrieval-Augmented Generation)** pipeline including:

- **Semantic Chunking**
- **BGE Embeddings**
- **Qdrant Vector Database**
- **Cross-Encoder Reranking**
- **Groq LLaMA-3.1 LLM for final answer generation**

It displays **Before vs After Reranking** chunks **side-by-side**.

---

## 🚀 Features

###  Upload any PDF  
Automatically extracts full text and splits into meaningful semantic chunks.

###  Vector Search (Before Reranking)  
Uses BGE-base embeddings stored in Qdrant (running via Docker).

###  Reranking (After Reranking)  
Uses **BAAI/bge-reranker-base** CrossEncoder.

###  LLM Answer Generation  
Powered by **Groq LLaMA-3.1-8B-Instant**.

###  Beautiful UI  
Chunk blocks displayed with pastel colors **side-by-side**.

---

##  Project Structure

search_pdf/  
│── app.py              # Streamlit UI  
│── search.py           # RAG pipeline (chunking, embeddings, Qdrant, LLM)  
│── requirements.txt    # All dependencies  
│── .env                # Your GROQ_API_KEY  
│── README.txt          # (this file)  

---

##  Qdrant Setup (Required)

Qdrant must run locally.

### 1. Install Docker  
```bash
https://www.docker.com/products/docker-desktop/
```

### 2. Pull Qdrant  
```bash

docker pull qdrant/qdrant
```


### 3. Run Qdrant  
```bash

docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant
```


-You should see Qdrant running at:  

http://localhost:6333


---
## Installation

1. **Clone the repository**
```bash
git clone <repository_url>

```

2. **Create virtual environment**
```bash
conda create -p venv python=3.12 -y
```
(Any other method to create a Python environment can also be used.)

3. **Activate environment**
```bash
conda activate ./venv
```
4.  **Environment Variables**

Create a `.env` file:  
```bash

GROQ_API_KEY=your_api_key_here

```

## How to run
### Step 1 — Install dependencies  

```bash
pip install -r requirements.txt

```


### Step 2 — Start Streamlit  
```bash

streamlit run app.py
```


### Step 3 — Upload PDF → Process → Ask a Question

---

## 🧠 Technologies Used

| Component     | Library / Model                  |
|---------------|----------------------------------|
| Embeddings    | BAAI/bge-base-en                 |
| Reranker      | BAAI/bge-reranker-base           |
| LLM           | LLaMA-3.1-8B-Instant (Groq)      |
| Vector DB     | Qdrant                           |
| UI            | Streamlit                        |
| Chunking      | LangChain SemanticChunker        |

---

## 📌 Notes

- **Qdrant must be running** before you search.  
- Requires a valid **Groq API Key**.  
- PDF extraction quality depends on PyPDF2.
--- 
## License

This project is licensed under the terms included in the LICENSE file.

---

## Author

**Anjali Bheemireddy**  
(anjalinature156@gmail.com)

