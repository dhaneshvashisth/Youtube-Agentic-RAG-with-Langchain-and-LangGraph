# YouTube Agentic RAG with LangChain & LangGraph

An **Agentic Retrieval-Augmented Generation (RAG) system** that allows users to **ask questions about any YouTube video**.  
The system automatically **extracts video transcripts, converts them into embeddings, stores them in a vector database, and uses an AI agent to generate context-aware answers.**

This project demonstrates **modern AI engineering practices using LangChain, LangGraph, and vector databases.**

---

# Features

- Extract transcripts from YouTube videos
- Chunk and preprocess transcripts
- Generate embeddings using OpenAI
- Store embeddings in **Qdrant vector database**
- Agent-based workflow using **LangGraph**
- Context-aware responses using **RAG**
- Semantic search across video transcripts
- Modular and scalable architecture

---

# Tech Stack

- Python
- LangChain
- LangGraph
- OpenAI API
- Qdrant Vector Database
- YouTube Transcript API
- python-dotenv

---

# Architecture

User Question  
↓  
LangGraph Agent  
↓  
Retrieve relevant transcript chunks from Qdrant  
↓  
LLM reasoning using LangChain  
↓  
Generate contextual response  

---

# Workflow

1. Extract transcript from YouTube video
2. Split transcript into smaller chunks
3. Generate embeddings
4. Store embeddings in Qdrant vector database
5. Retrieve relevant chunks based on the question
6. Generate answer using LLM with retrieved context

---

