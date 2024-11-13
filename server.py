from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import os

app = Flask(__name__)
CORS(app)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("Credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    if not all([name, email, password]):
        return jsonify({"error": "Missing required fields"}), 400
    try:
        db.collection('users').add({
            'name': name,
            'email': email,
            'password': password,
            'createdAt': firestore.SERVER_TIMESTAMP
        })
        return jsonify({"message": "User signed up successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    if not all([email, password]):
        return jsonify({"success": False, "message": "Missing required fields"}), 400
    try:
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).where('password', '==', password).get()
        if query:
            user = query[0].to_dict()
            return jsonify({"success": True, "user": {"name": user["name"], "email": user["email"]}}), 200
        else:
            return jsonify({"success": False, "message": "Invalid email or password"}), 401
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle
import google.generativeai as genai
from werkzeug.utils import secure_filename
import tempfile

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
TEMP_DIR = "temp_data"

# Create temporary directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer(MODEL_NAME)
        
    def extract_text_from_pdf(self, pdf_file):
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def chunk_text(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            if end > len(text):
                end = len(text)
            chunk = text[start:end]
            chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks
    
    def create_embeddings(self, chunks):
        return self.embedding_model.encode(chunks)

def initialize_faiss_index(dimension):
    return faiss.IndexFlatL2(dimension)

def setup_gemini():
    genai.configure(api_key="AIzaSyAOZRzjgX6LSv6FuG3pCmg-kmXJ8guYIdk")
    return genai.GenerativeModel('gemini-pro')

def save_data(index, texts):
    faiss.write_index(index, os.path.join(TEMP_DIR, "faiss_index.bin"))
    with open(os.path.join(TEMP_DIR, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

def load_data():
    index = faiss.read_index(os.path.join(TEMP_DIR, "faiss_index.bin"))
    with open(os.path.join(TEMP_DIR, "texts.pkl"), "rb") as f:
        texts = pickle.load(f)
    return index, texts

def get_relevant_chunks(query, index, texts, top_k=3):
    processor = DocumentProcessor()
    query_embedding = processor.embedding_model.encode([query])
    D, I = index.search(query_embedding, top_k)
    return [texts[i] for i in I[0]]

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    files = request.files.getlist('files')
    
    try:
        processor = DocumentProcessor()
        all_chunks = []
        
        # Process each PDF
        for file in files:
            if file.filename:
                text = processor.extract_text_from_pdf(file)
                chunks = processor.chunk_text(text)
                all_chunks.extend(chunks)
        
        # Create embeddings and initialize FAISS index
        embeddings = processor.create_embeddings(all_chunks)
        index = initialize_faiss_index(EMBEDDING_DIM)
        index.add(embeddings)
        
        # Save to disk
        save_data(index, all_chunks)
        
        return jsonify({'message': 'Files processed successfully'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        query_text = data.get('query')
        history = data.get('history', [])
        
        if not os.path.exists(os.path.join(TEMP_DIR, "faiss_index.bin")):
            return jsonify({'error': 'No documents processed yet'}), 400
        
        # Load data
        index, texts = load_data()
        
        # Get relevant chunks
        relevant_chunks = get_relevant_chunks(query_text, index, texts)
        context = "\n".join(relevant_chunks)
        
        # Prepare conversation history for context
        conversation_context = ""
        if history:
            conversation_context = "Previous conversation:\n"
            for conv in history[-3:]:  # Include last 3 conversations for context
                conversation_context += f"Q: {conv['question']}\nA: {conv['answer']}\n\n"
        
        # Setup Gemini and generate response
        model = setup_gemini()
        prompt = f"""Based on the following context and previous conversation history, please answer the question.
        If the answer cannot be found in the context, please say "I cannot find the answer in the provided documents."

        Document Context:
        {context}

        {conversation_context}
        Current Question: {query_text}"""
        
        response = model.generate_content(prompt)
        
        return jsonify({'answer': response.text}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

import base64

@app.route('/save-llm', methods=['POST'])
def save_llm():
    data = request.json
    llm_name = data.get('llmName')  # Changed from 'name' to 'llmName'
    user_email = data.get('userEmail')
    
    if not all([llm_name, user_email]):
        return jsonify({"error": "Missing required fields"}), 400
        
    try:
        # Read and encode the FAISS index
        with open(os.path.join(TEMP_DIR, "faiss_index.bin"), "rb") as f:
            faiss_data = base64.b64encode(f.read()).decode('utf-8')
            
        # Read and encode the texts file
        with open(os.path.join(TEMP_DIR, "texts.pkl"), "rb") as f:
            texts_data = base64.b64encode(f.read()).decode('utf-8')
            
        # Save to Firestore
        llm_doc = {
            'llmName': llm_name,  # Changed from 'name' to 'llmName'
            'userEmail': user_email,
            'faissIndex': faiss_data,
            'texts': texts_data,
            'createdAt': firestore.SERVER_TIMESTAMP
        }
        
        # Add to 'llms' collection
        db.collection('llms').add(llm_doc)
        
        return jsonify({"message": "LLM saved successfully"}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-user-llms', methods=['GET'])
def get_user_llms():
    user_email = request.args.get('email')
    if not user_email:
        return jsonify({"error": "Email parameter is required"}), 400
        
    try:
        # Query Firestore for LLMs associated with the user's email
        llms_ref = db.collection('llms')
        query = llms_ref.where('userEmail', '==', user_email).get()
        
        # Format the results
        llms = []
        for doc in query:
            llm_data = doc.to_dict()
            llms.append({
                'id': doc.id,
                'llmName': llm_data.get('llmName'),
                'createdAt': llm_data.get('createdAt')
            })
            
        return jsonify({"llms": llms}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/load-llm/<llm_id>', methods=['GET'])
def load_llm(llm_id):
    try:
        # Get the LLM document from Firestore
        llm_doc = db.collection('llms').document(llm_id).get()
        
        if not llm_doc.exists:
            return jsonify({"error": "LLM not found"}), 404
            
        llm_data = llm_doc.to_dict()
        
        # Decode base64 data
        faiss_data = base64.b64decode(llm_data['faissIndex'])
        texts_data = base64.b64decode(llm_data['texts'])
        
        # Save to temp directory
        with open(os.path.join(TEMP_DIR, "faiss_index.bin"), "wb") as f:
            f.write(faiss_data)
            
        with open(os.path.join(TEMP_DIR, "texts.pkl"), "wb") as f:
            f.write(texts_data)
            
        return jsonify({
            "message": "LLM loaded successfully",
            "llmName": llm_data['llmName']
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from flask import send_file

@app.route('/download-files/<file_type>', methods=['GET'])
def download_files(file_type):
    try:
        if file_type not in ['faiss', 'texts']:
            return jsonify({"error": "Invalid file type"}), 400
            
        file_path = os.path.join(TEMP_DIR, 
            "faiss_index.bin" if file_type == 'faiss' else "texts.pkl")
            
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
            
        return send_file(
            file_path,
            as_attachment=True,
            download_name=os.path.basename(file_path)
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/unlearn-data', methods=['POST'])
def unlearn_data():
    try:
        data = request.json
        text_to_unlearn = data.get('text')
        llm_id = data.get('llmId')
        user_email = data.get('userEmail')
        
        if not all([text_to_unlearn, llm_id, user_email]):
            return jsonify({"error": "Missing required fields"}), 400
            
        # Load current FAISS index and texts
        index, texts = load_data()
        
        # Create document processor for embeddings
        processor = DocumentProcessor()
        
        # Find similar chunks to remove
        text_embedding = processor.embedding_model.encode([text_to_unlearn])
        D, I = index.search(text_embedding, len(texts))  # Search all texts
        
        # Remove similar chunks (you can adjust the similarity threshold)
        similarity_threshold = 0.8
        chunks_to_keep = []
        embeddings_to_keep = []
        
        # Get embeddings for all texts
        all_embeddings = processor.embedding_model.encode(texts)
        
        for i, text in enumerate(texts):
            # Calculate cosine similarity
            similarity = np.dot(all_embeddings[i], text_embedding[0]) / (
                np.linalg.norm(all_embeddings[i]) * np.linalg.norm(text_embedding[0]))
            
            if similarity < similarity_threshold:
                chunks_to_keep.append(text)
                embeddings_to_keep.append(all_embeddings[i])
        
        # Create new FAISS index
        new_index = initialize_faiss_index(EMBEDDING_DIM)
        new_index.add(np.array(embeddings_to_keep))
        
        # Save updated data
        save_data(new_index, chunks_to_keep)
        
        # Update in Firebase
        with open(os.path.join(TEMP_DIR, "faiss_index.bin"), "rb") as f:
            faiss_data = base64.b64encode(f.read()).decode('utf-8')
            
        with open(os.path.join(TEMP_DIR, "texts.pkl"), "rb") as f:
            texts_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Update Firestore document
        db.collection('llms').document(llm_id).update({
            'faissIndex': faiss_data,
            'texts': texts_data,
            'updatedAt': firestore.SERVER_TIMESTAMP
        })
        
        return jsonify({"message": "Data successfully unlearned"}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
