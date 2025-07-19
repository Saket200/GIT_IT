from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyD_GDKAQ6uG9FtuvQ92qzHpa4mqIUV79Po"


st.header("GITRAG")

#THE RETREIVE PART


import requests
import os

def fetch_repo_content(repo_url, token=None):
    """
    Fetch content from a GitHub repository.
    
    Args:
        repo_url (str): URL of the GitHub repository (e.g., https://github.com/user/repo)
        token (str, optional): GitHub Personal Access Token for authentication
    
    Returns:
        list: List of dictionaries with file names and their content
    """
    # Extract owner and repo name from URL
    repo_path = repo_url.replace("https://github.com/", "").strip("/").split("/")
    if len(repo_path) < 2:
        raise ValueError("Invalid GitHub repository URL")
    owner, repo = repo_path[0], repo_path[1]
    
    # Set up headers for authentication
    headers = {"Authorization": f"token {token}"} if token else {}
    
    # Fetch repository contents
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()  # Raise an error for failed requests
    
    contents = response.json()
    documents = []
    
    # Process each file
    for item in contents:
        if item["type"] == "file" and item["name"].endswith((".md", ".py", ".txt")):  # Filter file types
            file_url = item["download_url"]
            if file_url:
                file_response = requests.get(file_url)
                file_response.raise_for_status()
                documents.append({
                    "name": item["name"],
                    "content": file_response.text
                })
    
    return documents




# Example: Fetch content from a public repository
repo_url = "https://github.com/langchain-ai/rag-from-scratch"
token = os.getenv("GITHUB_TOKEN")  # Set GITHUB_TOKEN in your environment
documents = fetch_repo_content(repo_url, token)

# Print fetched files
#for doc in documents:
  #  print(f"File: {doc['name']}\nContent (first 100 chars): {doc['content'][:1000]}...\n")


#Chunking and splitting

#splitter = RecursiveCharacterTextSplitter(
    #chunk_size=1000,
    #chunk_overlap=200,

#)

#chunks = splitter.create_documents([doc["content"] for doc in documents])

#print(chunks[1].page_content)


## EMBEDDING GENERATION AND VECTOR STORAGE



def process_and_store_docs(documents):
    """
    Split documents into chunks and store embeddings in FAISS.
    
    Args:
        documents (list): List of dictionaries with file names and content.
    
    Returns:
        FAISS: Vector store with document embeddings.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    texts = []
    metadatas = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        texts.extend(chunks)
        metadatas.extend([{"source": doc["name"]}] * len(chunks))
    
    # Create FAISS vector store
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore

    
    
    
#xmail = process_and_store_docs(documents)
#print(xmail.index_to_docstore_id)




# Create RAG chain with ChatGoogleGenerativeAI
def create_rag_chain(vector_store):
    """
    Create a RetrievalQA chain using ChatGoogleGenerativeAI.
    
    Args:
        vector_store (FAISS): FAISS vector store with document embeddings.
    
    Returns:
        RetrievalQA: RAG chain for answering queries.
    """
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    prompt_template = """Use the following context to answer the question. Provide a clear and concise explanation. If you don't know the answer, say so. Always end with "Thanks for asking!"

Context: {context}

Question: {question}
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

import asyncio


def main():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    repo_url = st.text_input("Enter GitHub Repository URL: ")
    token = os.getenv("GITHUB_TOKEN")  # Fine-Grained or Classic PAT
    try:
        # Fetch and process repository content
        documents = fetch_repo_content(repo_url, token)
        if not documents:
            print("No relevant files found in the repository.")
            return
        
        vector_store = process_and_store_docs(documents)
        rag_chain = create_rag_chain(vector_store)
        
        # Interactive chat loop
        while True:
            query = st.text_input("Ask a question about the repository (or 'exit' to quit): ")
            if query.lower() == "exit":
                break
            response = rag_chain.invoke(query)
            if st.button('summarize'): 
             st.write(response)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
     main()


    