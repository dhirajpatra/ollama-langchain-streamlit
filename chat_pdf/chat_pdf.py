from flask import Flask, request, jsonify
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers.document_compressors import LLMChainExtractor

import os
import uuid
import redis


app = Flask(__name__)

# Connect to Redis
redis_client = redis.from_url('redis://redis:6379/0')

class ChatPDF:
    
    def __init__(self):
        self.model = ChatOllama(model="phi3:latest", base_url="http://ollama:11434", verbose=True)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = None
        self.prompt_template = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for answering questions. Use the following context to answer the question. 
            If you do not know the answer, simply say that you do not know. Use at most three sentences and be concise in your response. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        try:
            # Generate a unique ID for the uploaded PDF
            pdf_id = str(uuid.uuid4())

            # Load and process the PDF
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            chunks = self.text_splitter.split_documents(docs)
            print(f"chunks created: {len(chunks)}")

            # Create a session-based directory
            session_id = request.form.get('session_id', str(uuid.uuid4()))
            session_dir = os.path.join('/app/uploads', session_id)
            os.makedirs(session_dir, exist_ok=True)
            print(f"session dir created: {session_dir}")

            # Create and persist the Chroma DB
            persist_directory = f"./chroma_db/{session_id}"
            os.makedirs(persist_directory, exist_ok=True)
            self.db = Chroma.from_documents(chunks, self.embedding, persist_directory=persist_directory)
            print(f"persist chroma db created: {self.db}")

            # Save the PDF
            pdf_save_path = os.path.join(session_dir, pdf_id + '.pdf')
            with open(pdf_save_path, 'wb') as f:
                f.write(open(pdf_file_path, 'rb').read())
            print(f"pdf saved")

            # Store session data in Redis
            session_data = {'pdf_path': pdf_save_path, 'embeddings_path': persist_directory}
            redis_client.hset(session_id, mapping=session_data)
            print(f"redis client mapped session data: {session_data}")

            return jsonify({"session_id": session_id, "message": "PDF ingested successfully"})
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

    
    def ask(self, session_id: str, query: str):
        try:
            cache_key = f"{session_id}:{query}"
            cached_response = redis_client.get(cache_key)
            if cached_response:
                return jsonify({"response": cached_response.decode('utf-8')})
            print(f"Cached response: {cached_response}")

            session_data = redis_client.hgetall(session_id)
            if not session_data:
                return jsonify({"error": "Session not found. Please upload a PDF document first."}), 404
            print(f"Session data: {session_data}")

            persist_directory = session_data.get(b'embeddings_path')
            if persist_directory is None:
                raise ValueError("Persist directory not found in session data.")
            persist_directory = persist_directory.decode('utf-8')

            if not os.path.exists(persist_directory):
                raise FileNotFoundError(f"Persist directory {persist_directory} does not exist.")

            self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
            print("Chroma DB initialized.")

            # Perform similarity search
            matching_docs = self.db.similarity_search(query)
            print(f"{len(matching_docs)} matching documents/results found")

            if not matching_docs:
                return jsonify({"response": "No relevant documents/result found."})

            # Set up the chain with the retrieved documents
            chain = RetrievalQA.from_chain_type(
                llm=self.model,
                retriever=self.db.as_retriever(),
                chain_type='stuff',
                chain_type_kwargs={"prompt": self.prompt_template},
                return_source_documents=True,
                verbose=True
            )
            print("Chain created.")

            # Run the chain on the matching documents
            result = chain.invoke({"input_documents": matching_docs, "query": query})
            response_text = result.get('result', 'No answer found.')
            print(f"Response text: {response_text}")

            # Cache the response
            redis_client.set(cache_key, response_text)
            print("Response cached.")

            return jsonify({"response": response_text})
        except Exception as e:
            # Enhanced error logging
            print(f"Error during query processing: {str(e)}")
            return jsonify({"error": f"Error during query processing: {str(e)}"}), 500
        

@app.route('/ingest', methods=['POST'])
def ingest():
    session_id = request.form.get('session_id', str(uuid.uuid4()))
    pdf_file = request.files['file']
    temp_file_path = f'/tmp/{pdf_file.filename}'
    pdf_file.save(temp_file_path)

    chat_pdf = ChatPDF()
    result = chat_pdf.ingest(temp_file_path)
    os.remove(temp_file_path)

    return result

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    session_id = data.get('session_id')
    query = data.get('query')

    chat_pdf = ChatPDF()
    response = chat_pdf.ask(session_id, query)
    return response

@app.route('/clear', methods=['POST'])
def clear():
    data = request.json
    session_id = data.get('session_id')

    if redis_client.exists(session_id):
        redis_client.delete(session_id)
        return jsonify({"message": "Session cleared"})
    else:
        return jsonify({"message": "Session not found"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
