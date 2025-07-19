"""
Example usage of the enhanced GitHub RAG system with output parsers

This script demonstrates how to use the different modes:
1. Code Analysis
2. Code Generation  
3. Code Refactoring
4. Explanation

Run this script to see how the output parsers work with structured responses.
"""

import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from typing import List

# Set your API key
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"

# Example Pydantic models (same as in gitrag.py)
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

def example_code_analysis():
    """Example of code analysis with output parser"""
    print("=== Code Analysis Example ===")
    
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    parser = PydanticOutputParser(pydantic_object=CodeAnalysis)
    
    prompt = PromptTemplate(
        template="Analyze this code and provide a structured analysis:\n\n{code}\n\n{format_instructions}",
        input_variables=["code"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Example code to analyze
    sample_code = """
def process_user_data(users):
    results = []
    for user in users:
        if user.get('active'):
            processed = {
                'id': user['id'],
                'name': user['name'].upper(),
                'email': user['email'].lower()
            }
            results.append(processed)
    return results
    """
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"code": sample_code})
        print(f"Summary: {result.summary}")
        print(f"Key Functions: {result.key_functions}")
        print(f"Dependencies: {result.dependencies}")
        print(f"Complexity: {result.complexity}")
        print(f"Suggestions: {result.suggestions}")
    except Exception as e:
        print(f"Error: {e}")

def example_code_generation():
    """Example of code generation with output parser"""
    print("\n=== Code Generation Example ===")
    
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    parser = PydanticOutputParser(pydantic_object=CodeGeneration)
    
    prompt = PromptTemplate(
        template="Generate code for: {requirement}\n\n{format_instructions}",
        input_variables=["requirement"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    requirement = "Create a function to validate email addresses with proper regex"
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"requirement": requirement})
        print(f"Filename: {result.filename}")
        print(f"Code:\n{result.code}")
        print(f"Explanation: {result.explanation}")
        print(f"Dependencies: {result.dependencies}")
        print(f"Usage Example:\n{result.usage_example}")
    except Exception as e:
        print(f"Error: {e}")

def example_with_context():
    """Example showing how context from repository would be used"""
    print("\n=== Context-Aware Code Generation ===")
    
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    parser = PydanticOutputParser(pydantic_object=CodeGeneration)
    
    # Simulated context from repository
    context = """
    # Example repository context
    import streamlit as st
    import requests
    from langchain.chains import RetrievalQA
    
    def fetch_data(url):
        response = requests.get(url)
        return response.json()
    
    def process_data(data):
        return [item for item in data if item['active']]
    """
    
    prompt = PromptTemplate(
        template="Based on this context:\n{context}\n\nGenerate code for: {requirement}\n\nFollow the same patterns and style.\n\n{format_instructions}",
        input_variables=["context", "requirement"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    requirement = "Create a function to save processed data to a file"
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "context": context,
            "requirement": requirement
        })
        print(f"Generated Code:\n{result.code}")
        print(f"Explanation: {result.explanation}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Make sure to set your API key
    if os.environ.get("GOOGLE_API_KEY") == "YOUR_API_KEY_HERE":
        print("Please set your GOOGLE_API_KEY environment variable")
        exit(1)
    
    # Run examples
    example_code_analysis()
    example_code_generation()
    example_with_context()
    
    print("\n=== Usage Instructions ===")
    print("1. Set GOOGLE_API_KEY environment variable")
    print("2. Run the Streamlit app: streamlit run gitrag.py")
    print("3. Choose different modes in the UI:")
    print("   - Explanation: Ask questions about code")
    print("   - Code Analysis: Get structured analysis")
    print("   - Code Generation: Generate new code")
    print("   - Code Refactoring: Improve existing code") 