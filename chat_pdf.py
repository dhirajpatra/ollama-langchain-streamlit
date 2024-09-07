# chat_pdf.py

import os
import uuid
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv(".env.dev")

app = Flask(__name__)

# Global variables for admin-ingested RAG data
global_rag_data = None
session_id = "abcd12345678"  # This ID can be dynamic if needed
persist_directory = "/app/chroma_db/global"

# Access the OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PROJECT_ID = os.getenv('PROJECT_ID')
ORGANIZATION_ID = os.getenv('ORGANIZATION_ID')
ORGANIZATION_NAME = os.getenv('ORGANIZATION_NAME')

class ChatPDF:
    
    def __init__(self):
        # Initialize the OpenAI Chat model
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=OPENAI_API_KEY,
            organization=ORGANIZATION_ID,
            temperature=0.2
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        # create open ai embedding
        # self.embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
        # create the open-source embedding function
        self.embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.prompt_template = PromptTemplate.from_template(
            """
            <s> [INST] You are an expert assistant for a Low-Code/No-Code (LCNC) platform, designed to guide users in developing applications. 
            Use the context provided below to answer the user's question with specific reference to how they can use the LCNC platform to achieve their goals. 
            If the context doesn't provide enough information, state that clearly. Keep your response concise and informative. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )


    def ingest(self, pdf_file_path: str, session_id: str):
        try:
            # Check if the embedding already exists
            if os.path.exists(persist_directory) and os.listdir(persist_directory):
                print(f"Embedding already exists in {persist_directory}. Loading existing data...")
                db = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
            else:
                # If embedding doesn't exist, initialize the database
                docs = PyPDFLoader(file_path=pdf_file_path).load()
                chunks = self.text_splitter.split_documents(docs)
                print(f"New embedding chunks created: {len(chunks)}")

                os.makedirs(persist_directory, exist_ok=True)
                db = Chroma.from_documents(chunks, self.embedding, persist_directory=persist_directory)
                print(f"FAQ RAG data created and persisted in {persist_directory}.")


            # Update global RAG data
            global global_rag_data
            global_rag_data = db  # Update the global RAG with the latest ingested data

            return jsonify({"session_id": session_id, "message": "PDF ingested successfully and FAQ data created."})
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

    def ask(self, session_id: str, query: str):
        try:
            # Use global RAG if session-specific data is not available
            db = global_rag_data
            if db is None:
                return jsonify({"error": "FAQ data is not available. Please contact the admin to upload the FAQ document."}), 404

            matching_docs = db.similarity_search(query)
            print(f"Matched documents: {len(matching_docs)}")

            if not matching_docs:
                return jsonify({"response": "No relevant documents/result found."})

            chain = RetrievalQA.from_chain_type(
                llm=self.model,
                retriever=db.as_retriever(),
                chain_type='stuff',
                chain_type_kwargs={"prompt": self.prompt_template},
                return_source_documents=True,
                verbose=True
            )
            result = chain.invoke({"input_documents": matching_docs, "query": query})
            response_text = result.get('result', 'No answer found.')

            return jsonify({"response": response_text})
        except Exception as e:
            print(f"Error during query processing: {str(e)}")
            return jsonify({"error": f"Error during query processing: {str(e)}"}), 500

@app.route('/ingest', methods=['POST'])
def admin_ingest():
    try:
        pdf_file = request.files['file']
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        temp_file_path = f'/tmp/{pdf_file.filename}'
        pdf_file.save(temp_file_path)

        chat_pdf = ChatPDF()
        result = chat_pdf.ingest(temp_file_path, session_id)
        os.remove(temp_file_path)

        return result
    except Exception as e:
        print(f"Error ingesting PDF: {str(e)}")
        return jsonify({"error": f"Error ingesting PDF: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        session_id = data.get('session_id')
        query = data.get('query')
        
        if not session_id:
            return jsonify({"error": "Session ID is not set. Please ask the admin to upload the relevant documents."}), 400

        print(f"ask API parameters - Session ID: {session_id}, Query: {query}")

        chat_pdf = ChatPDF()
        response = chat_pdf.ask(session_id, query)
        return response
    except Exception as e:
        print(f"Error during query processing: {str(e)}")
        return jsonify({"error": f"Error during query processing: {str(e)}"}), 500

if __name__ == '__main__':
    # Initialize global RAG data if exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        # embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        global_rag_data = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        print("Global RAG data loaded successfully.")

    app.run(host='0.0.0.0', port=8000)
