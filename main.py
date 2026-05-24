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

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# API KEY
load_dotenv(override=True)
google_api_key = (os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

os.environ["GOOGLE_API_KEY"] = google_api_key

video_id = "Gfr50f6ZBvo"


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


# =========================
# STEP 1: GET TRANSCRIPT
# =========================
try:
    video_metadata = get_video_metadata(video_id)
    ytt_api = YouTubeTranscriptApi(http_client=create_session())
    transcript_list = ytt_api.fetch(video_id, languages=["en", "en-US", "en-IN", "hi"])
    transcript = " ".join(chunk.text for chunk in transcript_list)

except TranscriptsDisabled:
    raise ValueError("No captions available for this video.")

# =========================
# STEP 2: SPLIT TEXT
# =========================

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.create_documents([video_metadata, transcript])

# =========================
# STEP 3: EMBEDDINGS + FAISS
# =========================
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# =========================
# STEP 4: LLM
# =========================
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

# =========================
# STEP 5: PROMPT
# =========================
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided video metadata and transcript context.
If the context is insufficient, just say you don't know.
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
# STEP 6: FORMAT FUNCTION
# =========================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# =========================
# STEP 7: CHAINS
# =========================
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

# =========================
# STEP 8: RUN
# =========================
question = "Can you summarize the video?"

response = main_chain.invoke(question)

print("\nAnswer:\n", response)
