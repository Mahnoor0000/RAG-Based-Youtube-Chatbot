
# RAG-Based YouTube Chatbot

A simple YouTube chatbot that allows users to ask questions about a YouTube video using its transcript.

This project uses Retrieval-Augmented Generation (RAG). It fetches the transcript of a YouTube video, converts the transcript into embeddings, stores them in a FAISS vector store, and then uses a Groq LLM to answer questions based only on the video content.

---

## Features

- Enter a YouTube video URL or video ID
- Fetch transcript from the video
- Split transcript into smaller chunks
- Generate embeddings using Hugging Face Sentence Transformers
- Store embeddings using FAISS
- Retrieve relevant transcript chunks for each question
- Generate answers using Groq Chat model
- Simple Streamlit user interface

---

## Technologies Used

- Python
- Streamlit
- LangChain
- Hugging Face Sentence Transformers
- FAISS
- Groq API
- YouTube Transcript API

---

## How It Works

1. The user enters a YouTube video URL or video ID.
2. The app extracts the video ID.
3. The transcript is fetched using YouTube Transcript API.
4. The transcript is divided into smaller text chunks.
5. Each chunk is converted into embeddings using a Hugging Face model.
6. The embeddings are stored in a FAISS vector database.
7. When the user asks a question, the most relevant transcript chunks are retrieved.
8. Groq LLM generates an answer using only the retrieved video context.

---

## Important Note

This app works only for YouTube videos that have captions or transcripts available.

If a video does not have captions, the transcript cannot be fetched.

The app currently works best when running locally. On cloud platforms, YouTube may block transcript requests from server IP addresses.


```
