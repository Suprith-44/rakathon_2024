import streamlit as st
import faiss
import pickle
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Initialize constants
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

class RAGChatbot:
    def __init__(self):
        self.embedding_model = SentenceTransformer(MODEL_NAME)
        self.gemini_model = None
        self.index = None
        self.texts = None
        
    def setup_gemini(self, api_key):
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
    def load_data(self, faiss_path, texts_path):
        # Load FAISS index
        self.index = faiss.read_index(faiss_path)
        
        # Load texts
        with open(texts_path, 'rb') as f:
            self.texts = pickle.load(f)
            
    def get_relevant_chunks(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(query_embedding, top_k)
        return [self.texts[i] for i in I[0]]
        
    def generate_response(self, query, history):
        try:
            # Get relevant chunks
            relevant_chunks = self.get_relevant_chunks(query)
            context = "\n".join(relevant_chunks)
            
            # Prepare conversation history
            conversation_context = ""
            if history:
                conversation_context = "Previous conversation:\n"
                for q, a in history[-3:]:  # Include last 3 conversations
                    conversation_context += f"Q: {q}\nA: {a}\n\n"
            
            # Generate prompt
            prompt = f"""Based on the following context and previous conversation history, please answer the question.
            If the answer cannot be found in the context, please say "I cannot find the answer in the provided documents."

            Document Context:
            {context}

            {conversation_context}
            Current Question: {query}"""
            
            # Generate response
            response = self.gemini_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    
    st.title("📚 RAG Chatbot")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
        st.session_state.is_initialized = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File uploaders
        faiss_file = st.file_uploader("Upload FAISS Index (.bin)", type=['bin'])
        texts_file = st.file_uploader("Upload Texts (.pkl)", type=['pkl'])
        
        # API key input
        api_key = st.text_input("Enter Gemini API Key", type="password")
        
        # Initialize button
        if st.button("Initialize Chatbot"):
            if not all([faiss_file, texts_file, api_key]):
                st.error("Please provide all required files and API key")
            else:
                try:
                    # Save uploaded files temporarily
                    with open("temp_faiss.bin", "wb") as f:
                        f.write(faiss_file.getbuffer())
                    with open("temp_texts.pkl", "wb") as f:
                        f.write(texts_file.getbuffer())
                    
                    # Initialize chatbot
                    st.session_state.chatbot.setup_gemini(api_key)
                    st.session_state.chatbot.load_data("temp_faiss.bin", "temp_texts.pkl")
                    st.session_state.is_initialized = True
                    
                    # Clean up temporary files
                    os.remove("temp_faiss.bin")
                    os.remove("temp_texts.pkl")
                    
                    st.success("Chatbot initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing chatbot: {str(e)}")
        
        if st.session_state.is_initialized:
            st.success("✅ Chatbot is ready!")
        else:
            st.warning("⚠️ Please initialize the chatbot")
    
    # Main chat interface
    if st.session_state.is_initialized:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for q, a in st.session_state.chat_history:
                st.write(f"👤 **You:** {q}")
                st.write(f"🤖 **Bot:** {a}")
                st.write("---")
        
        # Chat input
        with st.container():
            query = st.text_input("Ask a question:")
            if st.button("Send"):
                if query:
                    # Generate response
                    response = st.session_state.chatbot.generate_response(
                        query, 
                        st.session_state.chat_history
                    )
                    
                    # Update chat history
                    st.session_state.chat_history.append((query, response))
                    
                    # Use experimental_rerun instead of rerun
                    st.experimental_rerun()
    else:
        st.info("Please initialize the chatbot using the sidebar configuration.")

if __name__ == "__main__":
    main()