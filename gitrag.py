





import os
import streamlit as st
import requests
import threading
import asyncio
import nest_asyncio
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
nest_asyncio.apply()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field, ValidationError

# Set page configuration
st.set_page_config(page_title="GitHub Repository Chat & Code Generator", layout="wide")

# Output Parser Models
class CodeAnalysis(BaseModel):
    """Structured output for code analysis"""
    summary: str = Field(description="Brief summary of what the code does")
    key_functions: List[str] = Field(description="List of main functions/classes")
    dependencies: List[str] = Field(description="List of imports and dependencies")
    complexity: str = Field(description="Assessment of code complexity (low/medium/high)")
    suggestions: List[str] = Field(description="List of improvement suggestions")

class CodeGeneration(BaseModel):
    """Structured output for code generation"""
    filename: str = Field(description="Suggested filename for the generated code")
    code: str = Field(description="The complete generated code")
    explanation: str = Field(description="Explanation of what the code does")
    dependencies: List[str] = Field(description="Required imports and dependencies")
    usage_example: str = Field(description="Example of how to use the generated code")

class CodeRefactor(BaseModel):
    """Structured output for code refactoring"""
    original_code: str = Field(description="The original code that was analyzed")
    refactored_code: str = Field(description="The improved/refactored code")
    changes_made: List[str] = Field(description="List of changes made during refactoring")
    benefits: List[str] = Field(description="Benefits of the refactoring")

os.environ["GOOGLE_API_KEY"] = "##"
# Check for Google AI API key
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "last_repo_url" not in st.session_state:
    st.session_state.last_repo_url = ""
if "generated_files" not in st.session_state:
    st.session_state.generated_files = []

# Fetch repository content (synchronous)
def fetch_repo_content(repo_url, token=None):
    try:
        repo_path = repo_url.replace("https://github.com/", "").strip("/").split("/")
        if len(repo_path) < 2:
            raise ValueError("Invalid GitHub repository URL")
        owner, repo = repo_path[0], repo_path[1]
        headers = {"Authorization": f"token {token}"} if token else {}
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        contents = response.json()
        documents = []
        for item in contents:
            if item["type"] == "file" and item["name"].endswith((".md", ".py", ".txt", ".js", ".java", ".cpp", ".c", ".html", ".css")):
                file_url = item["download_url"]
                if file_url:
                    file_response = requests.get(file_url)
                    file_response.raise_for_status()
                    documents.append({
                        "name": item["name"],
                        "content": file_response.text,
                        "path": item["path"]
                    })
        return documents
    except requests.HTTPError as e:
        st.error(f"Error fetching repository: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return []

# Process documents and create FAISS vector store (synchronous)
def process_and_store_docs(documents):
    if not documents:
        return None
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        texts = []
        metadatas = []
        for doc in documents:
            chunks = text_splitter.split_text(doc["content"])
            texts.extend(chunks)
            metadatas.extend([{"source": doc["name"], "path": doc["path"]}] * len(chunks))
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        return vector_store
    except Exception as e:
        st.error(f"Error processing documents: {e}")
        return None

# Create different types of RAG chains with output parsers
def create_rag_chain(vector_store, chain_type="explanation"):
    try:
        llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
        
        if chain_type == "explanation":
            prompt_template = """Use the following context to answer the question. Provide a clear and concise explanation. If you don't know the answer, say so.

Context: {context}

Question: {question}

Answer:"""
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            parser = StrOutputParser()
            
        elif chain_type == "code_analysis":
            parser = PydanticOutputParser(pydantic_object=CodeAnalysis)
            prompt_template = """Analyze the following code context and provide a structured analysis.

Context: {context}

Question: {question}

{format_instructions}

Analysis:"""
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
        elif chain_type == "code_generation":
            parser = PydanticOutputParser(pydantic_object=CodeGeneration)
            prompt_template = """Based on the following context and requirements, generate code that follows the same patterns and style.

Context: {context}

Requirements: {question}

Generate code that:
1. Follows the coding patterns and conventions from the context
2. Uses similar libraries and dependencies
3. Has proper error handling and documentation
4. Is production-ready and well-structured

{format_instructions}

Generated Code:"""
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
        elif chain_type == "code_refactor":
            parser = PydanticOutputParser(pydantic_object=CodeRefactor)
            prompt_template = """Analyze the following code and provide a refactored version with improvements.

Context: {context}

Code to refactor: {question}

Provide a refactored version that:
1. Improves readability and maintainability
2. Follows best practices
3. Maintains the same functionality
4. Includes proper error handling

{format_instructions}

Refactored Code:"""
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return chain, parser
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        return None, None

# Run RAG query with output parsing
def run_rag_query(query, rag_chain, parser, result_container, chain_type="explanation"):
    def task():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = rag_chain({"query": query})
            loop.close()
            
            if chain_type == "explanation":
                result_container["response"] = result["result"]
            else:
                # Parse structured output
                parsed_result = parser.parse(result["result"])
                result_container["parsed_response"] = parsed_result
            
            result_container["sources"] = [doc.metadata["source"] for doc in result["source_documents"]]
        except ValidationError as e:
            result_container["error"] = f"Output parsing error: {e}"
        except Exception as e:
            result_container["error"] = str(e)

    thread = threading.Thread(target=task)
    thread.start()
    thread.join()

# Save generated code to file
def save_generated_code(filename, code_content):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code_content)
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

# Extract code blocks from markdown or text
def extract_code_blocks(text):
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
    return [block.strip() for block in code_blocks]

# Streamlit UI
st.title("GitHub Repository Chat & Code Generator")
st.write("Enter a GitHub repository URL and ask questions about its content, analyze code, or generate new code.")

# Repository input
repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/langchain-ai/rag-from-scratch")
token = st.text_input("GitHub Personal Access Token (optional)", type="password", value=os.getenv("GITHUB_TOKEN", ""))
process_button = st.button("Process Repository")

# Process repository
if process_button and repo_url:
    with st.spinner("Fetching and processing repository..."):
        documents = fetch_repo_content(repo_url, token)
        if documents:
            st.session_state.vector_store = process_and_store_docs(documents)
            if st.session_state.vector_store:
                st.session_state.last_repo_url = repo_url
                st.success(f"Processed repository: {repo_url}")
                st.info(f"Found {len(documents)} files to analyze")
            else:
                st.error("Failed to process documents.")
        else:
            st.error("No valid documents found.")

# Query section with different modes
if st.session_state.vector_store:
    st.subheader(f"Chat with {st.session_state.last_repo_url}")
    
    # Mode selection
    mode = st.selectbox(
        "Select Mode",
        ["Explanation", "Code Analysis", "Code Generation", "Code Refactoring"]
    )
    
    # Create appropriate chain based on mode
    chain_type_map = {
        "Explanation": "explanation",
        "Code Analysis": "code_analysis", 
        "Code Generation": "code_generation",
        "Code Refactoring": "code_refactor"
    }
    
    chain_type = chain_type_map[mode]
    rag_chain, parser = create_rag_chain(st.session_state.vector_store, chain_type)
    
    if rag_chain:
        # Different prompts based on mode
        if mode == "Explanation":
            query = st.text_area("Ask a question about the repository", height=100)
            placeholder = "e.g., What does this code do? How does the authentication work?"
        elif mode == "Code Analysis":
            query = st.text_area("Enter code to analyze or ask about specific code patterns", height=100)
            placeholder = "e.g., Analyze this function: def process_data(data): ..."
        elif mode == "Code Generation":
            query = st.text_area("Describe the code you want to generate", height=100)
            placeholder = "e.g., Create a function to validate email addresses, Create a class for user management"
        elif mode == "Code Refactoring":
            query = st.text_area("Enter code to refactor", height=100)
            placeholder = "e.g., def old_function(): ... (paste code to refactor)"
        
        st.text_area("Query", value=query, key=f"query_{mode}", placeholder=placeholder, height=100)
        
        if st.button(f"Submit {mode} Query"):
            if query:
                with st.spinner(f"Generating {mode.lower()}..."):
                    result_container = {}
                    run_rag_query(query, rag_chain, parser, result_container, chain_type)
                    
                    if "response" in result_container:
                        st.write("**Answer:**")
                        st.write(result_container["response"])
                        
                    elif "parsed_response" in result_container:
                        parsed = result_container["parsed_response"]
                        
                        if mode == "Code Analysis":
                            st.write("**Code Analysis:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Summary:**")
                                st.write(parsed.summary)
                                st.write("**Complexity:**")
                                st.write(parsed.complexity)
                            with col2:
                                st.write("**Key Functions:**")
                                for func in parsed.key_functions:
                                    st.write(f"- {func}")
                                st.write("**Dependencies:**")
                                for dep in parsed.dependencies:
                                    st.write(f"- {dep}")
                            
                            st.write("**Suggestions:**")
                            for suggestion in parsed.suggestions:
                                st.write(f"- {suggestion}")
                                
                        elif mode == "Code Generation":
                            st.write("**Generated Code:**")
                            st.code(parsed.code, language="python")
                            
                            st.write("**Explanation:**")
                            st.write(parsed.explanation)
                            
                            st.write("**Dependencies:**")
                            for dep in parsed.dependencies:
                                st.write(f"- {dep}")
                            
                            st.write("**Usage Example:**")
                            st.code(parsed.usage_example, language="python")
                            
                            # Save code option
                            if st.button("Save Generated Code"):
                                if save_generated_code(parsed.filename, parsed.code):
                                    st.session_state.generated_files.append(parsed.filename)
                                    st.success(f"Code saved to {parsed.filename}")
                                    
                        elif mode == "Code Refactoring":
                            st.write("**Original Code:**")
                            st.code(parsed.original_code, language="python")
                            
                            st.write("**Refactored Code:**")
                            st.code(parsed.refactored_code, language="python")
                            
                            st.write("**Changes Made:**")
                            for change in parsed.changes_made:
                                st.write(f"- {change}")
                                
                            st.write("**Benefits:**")
                            for benefit in parsed.benefits:
                                st.write(f"- {benefit}")
                            
                            # Save refactored code option
                            if st.button("Save Refactored Code"):
                                filename = f"refactored_{parsed.filename}"
                                if save_generated_code(filename, parsed.refactored_code):
                                    st.session_state.generated_files.append(filename)
                                    st.success(f"Refactored code saved to {filename}")
                    
                    st.write("**Sources:**")
                    for source in result_container["sources"]:
                        st.write(f"- {source}")
                        
                    if "error" in result_container:
                        st.error(f"Error generating response: {result_container['error']}")
            else:
                st.warning("Please enter a question.")

# Display generated files
if st.session_state.generated_files:
    st.subheader("Generated Files")
    for filename in st.session_state.generated_files:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
            with st.expander(f"📁 {filename}"):
                st.code(content, language="python")
                if st.button(f"Delete {filename}", key=f"delete_{filename}"):
                    os.remove(filename)
                    st.session_state.generated_files.remove(filename)
                    st.rerun()

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
## Modes:

**Explanation**: Ask questions about the repository content
**Code Analysis**: Get detailed analysis of code patterns and structure  
**Code Generation**: Generate new code based on repository patterns
**Code Refactoring**: Improve existing code with best practices

## Features:
- Structured output parsing for consistent results
- Code generation with proper formatting
- File saving and management
- Multi-language support (Python, JS, Java, etc.)

## Setup:
1. Set `GOOGLE_API_KEY` environment variable
2. Optionally set `GITHUB_TOKEN` for private repos
3. Enter repository URL and process
4. Choose mode and ask questions
""")


