from flask import Flask, request, jsonify
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers.document_compressors import LLMChainExtractor
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import uuid
import redis


app = Flask(__name__)

# Connect to Redis
redis_client = redis.from_url('redis://redis:6379/0')
llm_name = "codellama:7b-python-q5_K_S"
# llm_name = "phi3:latest"

class ChatPDF:
    
    def __init__(self):
        self.model = ChatOllama(model=llm_name, base_url="http://ollama:11434", verbose=True)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = None
        self.prompt_template = PromptTemplate.from_template(
            """
            <s> [INST] You are an advanced code generation assistant. Your task is to generate a complete Flask-based back-end application based on the provided front-end description extracted from a PDF.

            The front-end description will be dynamic and may include UI components, forms, and functionalities. Based on this description, generate the following:

            1. File Folder Structure: Outline the directory structure for the Flask application.
            2. Backend Files:
                - `app/__init__.py`: Initialize the Flask app and configure database connections.
                - `app/models.py`: Define database models based on the required fields.
                - `app/routes.py`: Implement API endpoints and handle logic for user registration and other functionalities.
                - `app/config.py`: Configure settings such as database URI.
                - `app/database.py`: Set up and initialize the database.
                - `migrations/env.py`: Configure Alembic migrations environment.
            3. Additional Files:
                - `requirements.txt`: List all necessary Python packages.
                - `Dockerfile`: Containerize the Flask application.
                - `docker-compose.yml`: Define services for Flask and the database.

            Ensure that:
            - The generated code aligns with the front-end functionalities described in the query.
            - The query is correctly structured and configured to work together.
            - Include any necessary explanations or setup instructions if required.

            Use the extracted details from the PDF as front end code to infer the necessary database fields, routes, and any other backend logic. Generate a complete and functional backend implementation.

            Context: {context}
            [/INST] </s>
            """
        )


    def save_embeddings_to_redis(self, session_id, chunk_text, embedding):
        try:
            if embedding is not None and len(embedding) > 0:
                embedding_str = ','.join(map(str, embedding))
                embedding_key = f"session:{session_id}:embedding:{chunk_text}"
                redis_client.set(embedding_key, embedding_str)
                print(f"Saved embedding for chunk: {chunk_text}")
            else:
                print(f"Empty embedding for chunk: {chunk_text}. Skipping save.")
        except Exception as e:
            print(f"Error saving embedding for chunk: {chunk_text}. Error: {str(e)}")


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

            # Save each chunk embedding in Redis
            for chunk in chunks:
                chunk_text = chunk.page_content
                chunk_embedding = self.embedding.embed_query(chunk_text)
                self.save_embeddings_to_redis(session_id, chunk_text, chunk_embedding)

                # Debugging information to check embedding type and content
                print(f"Chunk text: {chunk_text[:50]}")
                print(f"Chunk embedding type: {type(chunk_embedding)}")

                # Check if the embedding is a list and convert it to a numpy array
                if isinstance(chunk_embedding, list):
                    chunk_embedding = np.array(chunk_embedding)
                
                chunk_embedding_key = f"session:{session_id}:embedding:{chunk_text[:50]}"  # store a substring for the key
                redis_client.set(chunk_embedding_key, np.array2string(chunk_embedding, separator=','))
                print(f"Saved embedding for chunk: {chunk_text[:50]}")

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
            print(f"Session ID: {session_id}")
            print(f"cache key: {cache_key}")

            cached_response = redis_client.get(cache_key)
            if cached_response:
                return jsonify({"response": cached_response.decode('utf-8')})
            print("No cached response found")

            session_data = redis_client.hgetall(session_id)
            print(f"session data: {session_data}")
            if not session_data:
                return jsonify({"error": "Session not found. Please upload a PDF document first."}), 404

            persist_directory = session_data.get(b'embeddings_path')
            if persist_directory is None:
                raise ValueError("Persist directory not found in session data.")
            persist_directory = persist_directory.decode('utf-8')

            if not os.path.exists(persist_directory):
                raise FileNotFoundError(f"Persist directory {persist_directory} does not exist.")

            self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
            print("Chroma db object set")

            # Load KNN model and question embeddings
            self.load_redis_questions(session_id)
            print("Loaded questions from Redis")

            if self.knn_model:
                query_embedding = self.embedding.embed_query(query)
                print("Query embedding created")
                distances, indices = self.knn_model.kneighbors([query_embedding])
                print("KNN model called and found distances and indices")

                if distances[0][0] < 0.5:  # Assuming a distance threshold for similarity
                    similar_question = self.questions[indices[0][0]]
                    cached_response = redis_client.get(f"session:{session_id}:response:{similar_question}")
                    if cached_response:
                        return jsonify({"response": cached_response.decode('utf-8')})

            # Perform similarity search if no cached response
            matching_docs = self.db.similarity_search(query)
            print(f"Matched documents: {len(matching_docs)} numbers")

            if not matching_docs:
                return jsonify({"response": "No relevant documents/result found."})

            chain = RetrievalQA.from_chain_type(
                llm=self.model,
                retriever=self.db.as_retriever(),
                chain_type='stuff',
                chain_type_kwargs={
                    "prompt": self.prompt_template,
                    "document_variable_name": "context"
                },
                return_source_documents=True,
                verbose=True
            )
            print("Chain created")
            print(f"Query: {query}")
            print(f"Matching documents: {len(matching_docs)}")
            result = chain.invoke({"input_documents": matching_docs, "query": query, "context": query})
            response_text = result.get('result', 'No answer found.')
            print(f"Response text length: {len(response_text)}")

            # Cache the response and update KNN model
            redis_client.set(cache_key, response_text)
            self.update_redis_questions(session_id, query, response_text)
            print("Redis question set")

            return jsonify({"response": response_text})
        except Exception as e:
            print(f"Error during query processing: {str(e)}")
            return jsonify({"error": f"Error during query processing: {str(e)}"}), 500


    def load_redis_questions(self, session_id):
        try:
            question_keys = redis_client.keys(f"session:{session_id}:embedding:*")
            self.questions = []
            self.question_embeddings = []

            for key in question_keys:
                question_text = key.decode('utf-8').split(':', 3)[-1]
                self.questions.append(question_text)
                
                embedding_key = f"session:{session_id}:embedding:{question_text}"
                embedding_str = redis_client.get(embedding_key)
                print(f"Fetching embedding for: {embedding_key}")

                if embedding_str:
                    try:
                        embedding = np.fromstring(embedding_str.decode('utf-8'), sep=',')
                        
                        if embedding.size > 0:
                            self.question_embeddings.append(embedding)
                            print(f"Valid embedding loaded for: {embedding_key}")
                        else:
                            print(f"Empty embedding for: {embedding_key}")
                    except ValueError as ve:
                        print(f"Error parsing embedding for {embedding_key}: {ve}")
                else:
                    print(f"Embedding not found for: {embedding_key}")

            if self.question_embeddings:
                self.knn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
                self.knn_model.fit(self.question_embeddings)
                print("KNN model initialized.")
            else:
                self.knn_model = None
                print("No embeddings found to initialize KNN model.")
        except Exception as e:
            print(f"Error loading questions from Redis: {str(e)}")


    def update_redis_questions(self, session_id, question, response):
        try:
            question_embedding = self.embedding.embed_query(question)
            
            # Store the embedding with session-based key
            embedding_key = f"session:{session_id}:embedding:{question}"
            redis_client.set(embedding_key, np.array2string(question_embedding, separator=','))
            
            # Store the response with session-based key
            response_key = f"session:{session_id}:response:{question}"
            redis_client.set(response_key, response)
            
            # Reload questions and embeddings
            self.load_redis_questions(session_id)
            print("Redis questions and embeddings updated")
        except Exception as e:
            print(f"Error updating Redis: {str(e)}")

        
@app.route('/ingest', methods=['POST'])
def ingest():
    try:
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        pdf_file = request.files['file']
        temp_file_path = f'/tmp/{pdf_file.filename}'
        pdf_file.save(temp_file_path)

        chat_pdf = ChatPDF()
        result = chat_pdf.ingest(temp_file_path)
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
        print(f"ask API parameters Session ID: {session_id}, Query: {query}")

        chat_pdf = ChatPDF()
        response = chat_pdf.ask(session_id, query)
        return response
    except Exception as e:
        print(f"Error during query processing: {str(e)}")
        return jsonify({"error": f"Error during query processing: {str(e)}"}), 500
    

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