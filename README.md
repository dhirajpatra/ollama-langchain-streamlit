

## Ollama - Build a ChatBot with Langchain, Ollama & Deploy on Docker

This guide will walk you through the process of building a chatbot using Langchain and Ollama, and deploying it on Docker.

### Requirements

Before you start, make sure you have the following dependencies listed in your `requirements.txt` file:

- streamlit
- langchain
- langchain-community

### How to run

`docker-compose build --no-cache`
`docker-compose up`

Check both docker containers are running

`docker ps` 

Go to browser and check ollama http://localhost:11434

Now we need to pull the phi3 or other model by our ollama container. Get more details https://github.com/ollama/ollama and https://hub.docker.com/r/ollama/ollama

`docker exec -it ollama ollama run phi3`

Then check go to http://localhost:8501
