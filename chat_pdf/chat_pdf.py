# chat_pdf.py

from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
import os

app = Flask(__name__)

class ChatPDF:
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="phi3:latest", base_url="http://ollama:11434", verbose=True)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
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
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)

            # Initialize FastEmbedEmbeddings with the model name
            # we can use a parameter model_name="BAAI/bge-small-en-v1.5"
            embedding = FastEmbedEmbeddings()
            vector_store = Chroma.from_documents(documents=chunks, embedding=embedding)
            self.retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.5,
                },
            )

            self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                          | self.prompt
                          | self.model
                          | StrOutputParser())
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."
        return self.chain.invoke(query)

    def clear(self):
        self.retriever = None
        self.chain = None

# Initialize the ChatPDF instance
chat_pdf = ChatPDF()

@app.route('/ingest', methods=['POST'])
def ingest():
    pdf_file = request.files['file']
    temp_file_path = '/tmp/' + pdf_file.filename
    pdf_file.save(temp_file_path)
    result = chat_pdf.ingest(temp_file_path)
    os.remove(temp_file_path)  # Clean up the temporary file
    return jsonify({"message": result or "PDF ingested successfully"})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query')
    response = chat_pdf.ask(query)
    return jsonify({"response": response})

@app.route('/clear', methods=['POST'])
def clear():
    chat_pdf.clear()
    return jsonify({"message": "Cleared"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
