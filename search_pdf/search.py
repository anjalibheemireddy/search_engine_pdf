
# IMPORTS

import os
import nltk
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

from dotenv import load_dotenv
load_dotenv()
nltk.download("punkt")

semantic_embedder = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

bge = SentenceTransformer('BAAI/bge-base-en')
bge.max_seq_length = 512

reranker = CrossEncoder("BAAI/bge-reranker-base")

#qdrant = QdrantClient(path="qdrant_db")
collection_name = "rag_collection"

qdrant = QdrantClient(host="localhost", port=6333)


def init_collection():
    """Create collection if it does not exist."""
    if not qdrant.collection_exists("rag_collection"):
        qdrant.create_collection(
            collection_name="rag_collection",
            vectors_config=VectorParams(
                size=1024,    # BGE-large embedding size
                distance=Distance.COSINE
            ),
        )


groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def chunk_document(text):
    semantic_chunker = SemanticChunker(
        embeddings=semantic_embedder,
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=95
    )

    semantic_chunks = semantic_chunker.split_text(text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", "", "?"]
    )

    final_chunks = []
    for chunk in semantic_chunks:
        final_chunks.extend(recursive_splitter.split_text(chunk))

    return final_chunks

def embed_text(text_list):
    return bge.encode(text_list, normalize_embeddings=True).tolist()


def ingest_documents(texts):
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE,
        ),
    )

    points = []
    for idx, txt in enumerate(texts):
        vec = embed_text([txt])[0]
        points.append(
            PointStruct(
                id=idx,
                vector=vec,
                payload={"text": txt}
            )
        )

    qdrant.upsert(
        collection_name=collection_name,
        points=points
    )

def vector_search(query, top_k=10):
    vec = embed_text([query])[0]

    results = qdrant.query_points(
        collection_name=collection_name,
        query=vec,
        limit=top_k
    )

    docs = [{"text": p.payload.get("text", ""), "score_before": p.score}
            for p in results.points]

    return docs

def rerank_results(query, docs, top_k=5):
    pairs = [(query, d['text']) for d in docs]
    scores = reranker.predict(pairs)

    for d, s in zip(docs, scores):
        d["score_after"] = float(s)

    ranked = sorted(docs, key=lambda x: x["score_after"], reverse=True)
    return ranked[:top_k]

def build_prompt(question, passages):
    context = "\n".join([f"- {p['text']}" for p in passages])

    return f"""
You are a helpful assistant. Use ONLY the context below. If the answer is not in the context, respond with "I don't know".

### CONTEXT
{context}

### QUESTION
{question}

### ANSWER
"""

def answer_with_groq(question, passages):
    prompt = build_prompt(question, passages)

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content




def rag_query(question):
    docs = vector_search(question)
    reranked = rerank_results(question, docs)
    answer = answer_with_groq(question, reranked)

    return answer, docs, reranked
