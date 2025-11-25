# app.py
import streamlit as st
from PyPDF2 import PdfReader
from  search import chunk_document, ingest_documents, rag_query

st.set_page_config(page_title="📘 PDF Search Engine (RAG + Groq + BGE)", layout="wide")

st.title("📘 PDF Search Engine (RAG + Groq + BGE)")


# PDF UPLOAD & PROCESS

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    temp_path = "temp_uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    if st.button("Process PDF"):
        raw_text = ""
        reader = PdfReader(temp_path)
        for page in reader.pages:
            raw_text += page.extract_text() + "\n"

        chunks = chunk_document(raw_text)
        ingest_documents(chunks)
        st.success(f"PDF processed successfully! {len(chunks)} chunks stored in Qdrant.")



# SEARCH SECTION

st.subheader("🔍 Ask a Question")
query = st.text_input("Enter your question:")

if st.button("Search") and query.strip():
    # Run RAG pipeline
    answer, before_chunks, after_chunks = rag_query(query)

    st.write("### 💬 Answer")
    st.info(answer)

    
    # SIDE-BY-SIDE CHUNK DISPLAY
    
    st.markdown("---")
    st.subheader("📊 Chunks Considered")

    # Pastel colors
    before_colors = ["#FFFAE5", "#FFF0F5", "#E6FFFA", "#E0F7FA", "#F0F8FF"]
    after_colors = ["#E8F5E9", "#F1F8E9", "#E3F2FD", "#FFF3E0", "#FCE4EC"]

    col1, col2 = st.columns(2)

    # LEFT: Before Reranking
    with col1:
        st.write("###  Before Reranking")
        st.write(f"**Total retrieved:** {len(before_chunks)}")
        for i, chunk in enumerate(before_chunks):
            bg_color = before_colors[i % len(before_colors)]
            st.markdown(
                f"""
                <div style="
                    border:1px solid #D3D3D3;
                    border-radius:12px;
                    padding:10px;
                    margin-bottom:12px;
                    background-color:{bg_color};
                    height:140px;
                    overflow:hidden;
                ">
                    <b>Chunk {i+1}</b><br>
                    <span style="font-size:12px;">Score: {chunk['score_before']:.4f}</span>
                    <hr style="margin:4px 0;">
                    <div style="font-size:13px; height:90px; overflow-y:auto;">
                        {chunk['text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # RIGHT: After Reranking 
    with col2:
        st.write("###  After Reranking")
        st.write(f"**Final top selected:** {len(after_chunks)}")
        for i, chunk in enumerate(after_chunks):
            bg_color = after_colors[i % len(after_colors)]
            st.markdown(
                f"""
                <div style="
                    border:1px solid #BEE3BE;
                    border-radius:12px;
                    padding:10px;
                    margin-bottom:12px;
                    background-color:{bg_color};
                    height:140px;
                    overflow:hidden;
                ">
                    <b>Chunk {i+1}</b><br>
                    <span style="font-size:12px;">Rerank Score: {chunk['score_after']:.4f}</span>
                    <hr style="margin:4px 0;">
                    <div style="font-size:13px; height:90px; overflow-y:auto;">
                        {chunk['text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
