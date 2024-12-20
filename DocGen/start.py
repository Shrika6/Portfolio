import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from github import Github

# Load the open-source model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Function to extract functions from a Python code
def extract_functions_from_code(code):
    """
    Parses the code to extract function definitions.
    """
    tree = ast.parse(code)
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_info = {
                'name': node.name,
                'code': ast.unparse(node),  # Converts the AST node back to Python code
                'docstring': ast.get_docstring(node)
            }
            functions.append(func_info)
    
    return functions

# Function to generate documentation using an open-source model
def generate_doc(description):
    """
    Uses an open-source language model to generate documentation for a given code description.
    """
    prompt = f"Generate documentation for the following code:\n\n{description}\n\nDocumentation:"
    
    # Set the pad token to the EOS token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Use `max_new_tokens` for generation
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=150,  # Adjust this to control the length of generated output
        pad_token_id=tokenizer.pad_token_id
    )
    
    doc = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return doc.strip()



# Function to fetch files from a GitHub repository
def fetch_github_files(repo_name, file_extension=".py"):
    """
    Fetches files with a specific extension from a GitHub repository.
    """
    g = Github("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")
    repo = g.get_repo(repo_name)
    contents = repo.get_contents("")
    files = []

    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        elif file_content.path.endswith(file_extension):
            files.append(file_content)

    return files

# Example usage: Fetching files from a GitHub repo, extracting functions, and generating documentation
def document_github_repo(repo_name):
    files = fetch_github_files(repo_name)
    
    for file in files:
        file_content = file.decoded_content.decode('utf-8')
        functions = extract_functions_from_code(file_content)
        
        for func in functions:
            doc = generate_doc(func['code'])
            print(f"File: {file.path}")
            print(f"Function name: {func['name']}\nGenerated Documentation:\n{doc}\n")

# Example: Documenting a GitHub repository
if __name__ == "__main__":
    repo_name = "Shrika6/cnassignment"  # Replace with the name of the GitHub repo
    document_github_repo(repo_name)
