import streamlit as st
import re
import os
import requests
from dotenv import load_dotenv

for proxy_var in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ.pop(proxy_var, None)

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥")

st.markdown("## 🎥 YouTube RAG Chatbot")
st.markdown("Ask questions from any YouTube video using AI 🚀")

# =========================
# LOAD API KEY
# =========================
load_dotenv(override=True)
google_api_key = (os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key


def get_video_id(video_url):
    patterns = [
        r"(?:v=)([^&]+)",
        r"youtu\.be/([^?&]+)",
        r"youtube\.com/embed/([^?&]+)",
        r"youtube\.com/shorts/([^?&]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)

    return video_url.strip()


def create_session():
    session = requests.Session()
    session.trust_env = False
    return session


def get_video_metadata(video_id):
    response = create_session().get(
        "https://www.youtube.com/oembed",
        params={
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "format": "json",
        },
        timeout=10,
    )
    response.raise_for_status()
    metadata = response.json()

    return "\n".join(
        [
            f"Video title: {metadata.get('title', 'Unknown')}",
            f"YouTuber/channel name: {metadata.get('author_name', 'Unknown')}",
            f"Channel URL: {metadata.get('author_url', 'Unknown')}",
        ]
    )


def get_transcript(video_id):
    ytt_api = YouTubeTranscriptApi(http_client=create_session())
    transcript_list = ytt_api.fetch(
        video_id,
        languages=["en", "en-US", "en-IN", "hi"],
    )
    return " ".join(chunk.text for chunk in transcript_list)

# =========================
# INPUTS
# =========================
video_url = st.text_input("Enter YouTube Video URL:")
question = st.text_input("Ask a question about the video:")

# =========================
# BUTTON ACTION
# =========================
if st.button("Get Answer"):

    if not video_url or not question:
        st.warning("Please enter both video URL and question.")
    elif not google_api_key:
        st.error("GOOGLE_API_KEY is missing. Add it to your .env file.")
    else:
        with st.spinner("Processing video and generating answer..."):

            try:
                # =========================
                # EXTRACT VIDEO ID
                # =========================
                video_id = get_video_id(video_url)

                # =========================
                # GET TRANSCRIPT
                # =========================
                video_metadata = get_video_metadata(video_id)
                transcript = get_transcript(video_id)

                # =========================
                # SPLIT TEXT
                # =========================
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.create_documents([video_metadata, transcript])

                # =========================
                # EMBEDDINGS + FAISS
                # =========================
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})

                # =========================
                # LLM
                # =========================
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    temperature=0.2
                )

                # =========================
                # PROMPT
                # =========================
                prompt = PromptTemplate(
                    template="""
You are a helpful assistant.
Answer ONLY from the provided video metadata and transcript context.
If the context is insufficient, say "I don't know".
Always answer in the same language as the question. If the question is in
English, answer in English, even when the transcript is in another language.
You may translate facts from the context into the question's language.

Context:
{context}

Question:
{question}
""",
                    input_variables=["context", "question"]
                )

                # =========================
                # FORMAT FUNCTION
                # =========================
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # =========================
                # CHAIN
                # =========================
                chain = (
                    RunnableParallel({
                        "context": retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough()
                    })
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # =========================
                # GET ANSWER
                # =========================
                response = chain.invoke(question)

                st.success("Answer:")
                st.write(response)

            except TranscriptsDisabled:
                st.error("No captions available for this video.")

            except NoTranscriptFound:
                st.error("No supported transcript found for this video.")

            except Exception as e:
                error_message = str(e)
                if "API key expired" in error_message or "API_KEY_INVALID" in error_message:
                    st.error(
                        "Your GOOGLE_API_KEY is being rejected by Gemini. "
                        "Create a fresh Gemini API key, replace the value in .env, "
                        "then fully restart Streamlit."
                    )
                else:
                    st.error(f"Error: {error_message}")
