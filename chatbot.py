
# IMPORT LIBRARIES

import os
import re
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# EXTRACT VIDEO ID FROM URL

def extract_video_id(youtube_url_or_id):
    value = youtube_url_or_id.strip()

    # If user enters only video ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", value):
        return value

    parsed_url = urlparse(value)

    
    if "youtube.com" in parsed_url.netloc:
        query_params = parse_qs(parsed_url.query)

        if "v" in query_params:
            return query_params["v"][0]

        # Shorts / embed / live
        path_parts = parsed_url.path.strip("/").split("/")

        if len(path_parts) >= 2 and path_parts[0] in ["shorts", "embed", "live"]:
            return path_parts[1]
        
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.strip("/").split("?")[0]

    raise ValueError("Invalid YouTube URL or video ID")


# FETCH TRANSCRIPT

def get_video_transcript(youtube_url_or_id):
    video_id = extract_video_id(youtube_url_or_id)

    try:
        ytt_api = YouTubeTranscriptApi()

        # First try English transcript
        try:
            transcript_list = ytt_api.fetch(video_id, languages=["en", "en-US", "en-GB"])

        except Exception:
            # If English is not available, get first available transcript
            transcript_list_all = ytt_api.list(video_id)

            transcript_obj = None

            for transcript in transcript_list_all:
                transcript_obj = transcript
                break

            if transcript_obj is None:
                raise Exception("No transcript found for this video.")

            # Translate to English if possible
            try:
                if transcript_obj.is_translatable:
                    transcript_obj = transcript_obj.translate("en")
            except Exception:
                pass

            transcript_list = transcript_obj.fetch()

        transcript = " ".join(snippet.text for snippet in transcript_list)

        return transcript, video_id

    except TranscriptsDisabled:
        raise Exception("No captions/transcript available for this video.")

    except Exception as e:
        raise Exception(f"Could not fetch transcript: {e}")


# CREATE VECTOR STORE

def create_vector_store(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript])

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embedding_model)

    return vector_store, chunks


# PROCESS VIDEO

def process_video(youtube_url_or_id):
    transcript, video_id = get_video_transcript(youtube_url_or_id)

    vector_store, chunks = create_vector_store(transcript)

    return vector_store, transcript, video_id, len(chunks)



# FORMAT RETRIEVED DOCS

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


# ASK QUESTION

def ask_question(vector_store, question):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ.get("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say the video doesnt mention it.

Context:
{context}

Question: {question}
""",
        input_variables=["context", "question"]
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | llm | parser

    result = main_chain.invoke(question)

    return result