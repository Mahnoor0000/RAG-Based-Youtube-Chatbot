
import os
import streamlit as st
from dotenv import load_dotenv
from chatbot import process_video, ask_question


load_dotenv()


st.set_page_config(
    page_title="YouTube Chatbot",
    layout="centered"
)


st.title("🎬 YouTube Video Chatbot")

st.write(
    "Paste a YouTube video URL then ask questions about the video."
)

# YouTube URL input
youtube_url = st.text_input(
    "Enter YouTube Video URL or ID",
    placeholder="https://www.youtube.com/watch?v=9vM4p9NN0Ts"
)

index_button = st.button("Enter")

# Index video
if index_button:
    if not youtube_url.strip():
        st.error("Please enter a YouTube URL or video ID.")

    else:
        try:
            with st.spinner("Fetching transcript..."):
                vector_store, transcript, video_id, total_chunks = process_video(youtube_url)

            st.session_state["video_id"] = video_id
            st.session_state["transcript"] = transcript
            st.session_state["vector_store"] = vector_store
            st.session_state["total_chunks"] = total_chunks

            st.success("Video transcript fetched successfully!")

            
        except Exception as e:
            st.error(f"Could not process this video: {e}")



# Ask questions
if "vector_store" in st.session_state:
    st.markdown("---")
    st.subheader("Ask questions about this video")

    question = st.text_input(
        "Your question",
        placeholder="What is this video about?"
    )

    ask_button = st.button("Ask")

    if ask_button:
        if not question.strip():
            st.error("Please enter a question.")


        else:
            try:

                with st.spinner("Thinking..."):
                    answer = ask_question(
                        vector_store=st.session_state["vector_store"],
                        question=question
                    )

                st.markdown("### Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"Could not generate answer: {e}")

