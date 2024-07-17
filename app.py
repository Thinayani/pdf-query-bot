import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from main import split_pdf, get_Gemini_Response

# Streamlit App
st.title("Document Query Application")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    # Save the uploaded file
    with open("uploaded_document.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the PDF file
    st.write("Processing the PDF file...")
    pages = split_pdf("uploaded_document.pdf")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_index = FAISS.from_documents(pages, embedding_function)
    st.write("PDF processed successfully!")

    # User query
    query = st.text_area("Enter your query about the document:")
    if query:
        st.write("Searching for relevant information...")
        docs = faiss_index.similarity_search(query, k=4)
        pageList = []
        for doc in docs:
            pageList.append(doc.page_content)
        # prompt = f"For the following question: {query}, based on the document:\n {pageList}"
        prompt = (
            f"You are an expert on the story, possessing a comprehensive and nuanced understanding of its plot, "
            f"characters, themes, and intricate details. "
            f"Your task is to provide a thorough, clear, and accurate answer to the following question, "
            f"drawing exclusively from the context provided from the document excerpts. "
            f"Ensure that your response is well-informed and directly addresses the query.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n {pageList}"
        )

        # Get and display the response
        response = get_Gemini_Response(prompt)
        if response:
            st.write("Response from the model:")
            st.write(response)
        else:
            st.write("Failed to get a response from the model.")
