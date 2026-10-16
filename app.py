import streamlit as st
import os
import whisper
from rag_pipeline import build_vector_store, get_answer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Smart Video Lecture Assistant", page_icon="📚")
st.title("📚 Smart Video Lecture Assistant")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None

uploaded_video = st.file_uploader("Upload your lecture video", type=["mp4"])

if uploaded_video is not None:
    video_path = os.path.join("videos", "uploaded_lecture.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    if st.button("Process Video"):
        with st.spinner("Transcribing... (this may take a while)"):
            model = whisper.load_model("tiny")
            result = model.transcribe(video_path)
            transcript = result["text"]
            st.session_state.transcript = transcript

            transcript_path = "transcripts/uploaded_transcript.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)

        st.success("✅ Transcription complete!")

        if len(transcript.strip()) < 20:
            st.error("⚠️ The transcript is too short. Please try a longer video with clear audio (at least 1-2 minutes).")
        else:
            with st.spinner("Building knowledge base..."):
                vector_store = build_vector_store(transcript_path, persist_dir="vector_db_temp")
                st.session_state.vector_store = vector_store

            st.success("✅ Ready for questions!")

if st.session_state.transcript:
    with st.expander("📄 View Full Transcript"):
        st.write(st.session_state.transcript)

    st.subheader("📝 Lecture Summary")
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            max_chars = 12000
            text = st.session_state.transcript[:max_chars]
            prompt = "Create a topic-wise summary of the lecture transcript below (in bullet points):\n\n" + text
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            st.write(response.choices[0].message.content)

    st.subheader("💬 Ask a Question")
    query = st.text_input("Type your question:")
    if query and st.session_state.vector_store:
        with st.spinner("Searching for the answer..."):
            answer = get_answer(query, st.session_state.vector_store)
            st.write(answer)
    elif query and not st.session_state.vector_store:
        st.warning("⚠️ Please process the video first, then ask your question.")