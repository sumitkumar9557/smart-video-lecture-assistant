import os
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_vector_store(transcript_path, persist_dir="vector_db"):
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript_text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(chunks, embeddings, persist_directory=persist_dir)

    return vector_store


def load_vector_store(persist_dir="vector_db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vector_store


def search_relevant_chunks(query, vector_store, k=3):
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


def get_answer(query, vector_store):
    relevant_chunks = search_relevant_chunks(query, vector_store)
    context = "\n\n".join(relevant_chunks)

    instruction = (
        "Answer the student's question based on the lecture content given below. "
        "Answer only using the given context. If the answer is not present in the "
        "context, say: This information is not available in the lecture."
    )

    prompt = instruction