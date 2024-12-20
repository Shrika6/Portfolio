from flask import Flask, request, jsonify
import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from github import Github

app = Flask(__name__)

# Load the open-source model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Function to extract functions from a Python code
def extract_functions_from_code(code):
    tree = ast.parse(code)
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_info = {
                'name': node.name,
                'code': ast.unparse(node),
                'docstring': ast.get_docstring(node)
            }
            functions.append(func_info)
    
    return functions

# Function to generate documentation using an open-source model
def generate_doc(description):
    prompt = f"Generate documentation for the following code:\n\n{description}\n\nDocumentation:"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=150,
        pad_token_id=tokenizer.pad_token_id
    )
    
    doc = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return doc.strip()

# Function to fetch files from a GitHub repository
def fetch_github_files(repo_name, file_extension=".py"):
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

# Flask route to handle documentation requests
@app.route('/document', methods=['POST'])
def document_repo():
    data = request.json
    repo_name = data.get('repo_name')

    if not repo_name:
        return jsonify({'error': 'Repository name is required'}), 400

    try:
        files = fetch_github_files(repo_name)
        documentation = []

        for file in files:
            file_content = file.decoded_content.decode('utf-8')
            functions = extract_functions_from_code(file_content)
            
            for func in functions:
                doc = generate_doc(func['code'])
                documentation.append({
                    'file': file.path,
                    'function_name': func['name'],
                    'documentation': doc
                })
        
        return jsonify(documentation)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
