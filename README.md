# GitHub Repository Chat & Code Generator

An enhanced RAG (Retrieval-Augmented Generation) system that can read, analyze, and generate code from GitHub repositories using structured output parsers.

## 🚀 Key Features

### 1. **Output Parsers for Structured Responses**
- **Code Analysis**: Get detailed, structured analysis of code patterns
- **Code Generation**: Generate new code with proper formatting and documentation
- **Code Refactoring**: Improve existing code with best practices
- **Explanation**: Traditional Q&A about repository content

### 2. **Multi-Mode Operation**
- **Explanation Mode**: Ask questions about repository content
- **Code Analysis Mode**: Get structured analysis of code patterns and complexity
- **Code Generation Mode**: Generate new code based on repository patterns
- **Code Refactoring Mode**: Improve existing code with suggestions

### 3. **File Management**
- Save generated code to files
- View and manage generated files
- Delete files when no longer needed

### 4. **Multi-Language Support**
- Python, JavaScript, Java, C++, C, HTML, CSS, and more
- Language-specific code formatting

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Saket200/GIT_IT
cd gitrag
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
export GOOGLE_API_KEY="your-google-api-key"
export GITHUB_TOKEN="your-github-token"  # Optional
```

## 🎯 Usage

### Running the Streamlit App

```bash
streamlit run gitrag.py
```

### Using Different Modes

#### 1. **Explanation Mode**
- Ask general questions about the repository
- Get explanations of code functionality
- Understand project structure

**Example**: "What does this code do?" or "How does the authentication work?"

#### 2. **Code Analysis Mode**
- Get structured analysis of code patterns
- Identify key functions and dependencies
- Assess code complexity
- Receive improvement suggestions

**Example**: "Analyze this function: `def process_data(data): ...`"

#### 3. **Code Generation Mode**
- Generate new code based on repository patterns
- Follow existing coding conventions
- Include proper documentation and error handling
- Get usage examples

**Example**: "Create a function to validate email addresses"

#### 4. **Code Refactoring Mode**
- Improve existing code
- Follow best practices
- Maintain functionality while enhancing readability
- Get detailed change explanations

**Example**: Paste code to refactor and get improved version

## 📊 Output Parser Models

### CodeAnalysis
```python
class CodeAnalysis(BaseModel):
    summary: str                    # Brief summary of what the code does
    key_functions: List[str]        # List of main functions/classes
    dependencies: List[str]         # List of imports and dependencies
    complexity: str                 # Assessment of code complexity
    suggestions: List[str]          # List of improvement suggestions
```

### CodeGeneration
```python
class CodeGeneration(BaseModel):
    filename: str                   # Suggested filename for the generated code
    code: str                       # The complete generated code
    explanation: str                # Explanation of what the code does
    dependencies: List[str]         # Required imports and dependencies
    usage_example: str              # Example of how to use the generated code
```

### CodeRefactor
```python
class CodeRefactor(BaseModel):
    original_code: str              # The original code that was analyzed
    refactored_code: str            # The improved/refactored code
    changes_made: List[str]         # List of changes made during refactoring
    benefits: List[str]             # Benefits of the refactoring
```

## 🔧 Example Usage

### Programmatic Usage

```python
from example_usage import example_code_analysis, example_code_generation

# Run code analysis
example_code_analysis()

# Run code generation
example_code_generation()
```

### Streamlit Interface

1. **Enter Repository URL**: `https://github.com/username/repo`
2. **Choose Mode**: Select from Explanation, Code Analysis, Code Generation, or Code Refactoring
3. **Ask Question**: Enter your query based on the selected mode
4. **View Results**: Get structured responses with code formatting
5. **Save Code**: Optionally save generated code to files

## 🎨 Features in Detail

### Structured Output Parsing
- Uses Pydantic models for consistent, structured responses
- Validates output format automatically
- Provides clear error messages for parsing issues

### Context-Aware Code Generation
- Analyzes repository patterns and conventions
- Generates code that follows existing style
- Includes proper imports and dependencies

### File Management
- Save generated code with appropriate filenames
- View all generated files in expandable sections
- Delete files when no longer needed

### Error Handling
- Graceful handling of API errors
- Validation of output parsing
- Clear error messages for debugging

## 🔍 Advanced Features

### Repository Analysis
- Fetches multiple file types (`.py`, `.js`, `.java`, `.cpp`, etc.)
- Processes markdown documentation
- Maintains file path information for context

### Vector Storage
- Uses FAISS for efficient similarity search
- Stores code chunks with metadata
- Retrieves relevant context for code generation

### Multi-Threading
- Handles async operations properly
- Prevents UI blocking during processing
- Manages event loops correctly

## 🚨 Important Notes

1. **API Key**: Make sure to set your `GOOGLE_API_KEY` environment variable
2. **Rate Limits**: GitHub API has rate limits; use tokens for higher limits
3. **File Types**: Currently supports common programming languages and documentation
4. **Context Size**: Large repositories may take time to process

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GOOGLE_API_KEY` is set correctly
2. **Parsing Errors**: Check if the LLM response matches expected format
3. **File Save Errors**: Ensure write permissions in the current directory
4. **Repository Access**: Use GitHub token for private repositories

### Debug Mode

Enable debug output by checking the console for detailed error messages.

## 📈 Future Enhancements

- [ ] Support for more programming languages
- [ ] Integration with Git for version control
- [ ] Batch processing of multiple repositories
- [ ] Custom output parser templates
- [ ] Code testing and validation
- [ ] Integration with IDEs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 