from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import os

# API KEY
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

video_id = "Gfr50f6ZBvo"

# =========================
# STEP 1: GET TRANSCRIPT
# =========================
try:
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)

except TranscriptsDisabled:
    print("No captions available for this video.")

# =========================
# STEP 2: SPLIT TEXT
# =========================

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.create_documents([transcript])

# =========================
# STEP 3: EMBEDDINGS + FAISS
# =========================
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# =========================
# STEP 4: LLM
# =========================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

# =========================
# STEP 5: PROMPT
# =========================
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

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