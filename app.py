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
st.set_page_config(
    page_title=" RAG Chatbot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1120px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }

        .hero {
            padding: 1.25rem 0 1.75rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            font-size: 3rem;
            line-height: 1.05;
            margin: 0;
            letter-spacing: 0;
        }

        .hero p {
            margin: 0.75rem 0 0;
            color: #cbd5e1;
            font-size: 1.05rem;
        }

        .panel {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.42);
            border-radius: 8px;
            padding: 1.25rem;
        }

        .answer-card {
            border-left: 4px solid #22c55e;
            background: rgba(20, 83, 45, 0.24);
            border-radius: 8px;
            padding: 1.25rem 1.35rem;
            margin-top: 1rem;
        }

        .meta-line {
            color: #cbd5e1;
            font-size: 0.95rem;
            margin-bottom: 0.35rem;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 0.9rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 650;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🎥 YouTube RAG Chatbot</h1>
        <p>Ask focused questions from YouTube transcripts with Gemini-powered retrieval.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# LOAD API KEY
# =========================
load_dotenv(override=True)


def get_streamlit_secret(key):
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


google_api_key = (
    get_streamlit_secret("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_API_KEY", "")
).strip().strip('"').strip("'")
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


def parse_metadata(metadata_text):
    metadata = {}
    for line in metadata_text.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            metadata[key] = value
    return metadata


# =========================
# INPUTS
# =========================
with st.sidebar:
    st.markdown("### Project")
    st.markdown("YouTube transcript Q&A with metadata-aware retrieval.")
    st.divider()
    st.markdown("### Pipeline")
    st.markdown("1. Fetch video metadata")
    st.markdown("2. Load transcript")
    st.markdown("3. Build FAISS index")
    st.markdown("4. Answer with Gemini")
    st.divider()
    if google_api_key:
        st.success("Gemini key loaded")
    else:
        st.error("Gemini key missing")

left_col, right_col = st.columns([1.35, 0.85], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    with st.form("rag_form"):
        video_url = st.text_input(
            "YouTube video URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        question = st.text_input(
            "Question",
            placeholder="Example: What is this video about?",
        )
        submitted = st.form_submit_button("Get Answer")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("#### Try Asking")
    st.markdown(
        """
        - Who is the YouTuber?
        - Summarize the video.
        - What are the main points?
        - What does the speaker recommend?
        """
    )

# =========================
# BUTTON ACTION
# =========================
if submitted:

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
                metadata = parse_metadata(video_metadata)
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

                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="small-muted">Answer</div>
                        <div style="font-size: 1.08rem; margin-top: 0.35rem;">{response}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("Video Details", expanded=True):
                    st.markdown(
                        f"""
                        <div class="meta-line"><strong>Title:</strong> {metadata.get("Video title", "Unknown")}</div>
                        <div class="meta-line"><strong>Channel:</strong> {metadata.get("YouTuber/channel name", "Unknown")}</div>
                        <div class="meta-line"><strong>Video ID:</strong> {video_id}</div>
                        """,
                        unsafe_allow_html=True,
                    )

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
