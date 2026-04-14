import streamlit as st
import re
import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
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
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

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
    else:
        with st.spinner("Processing video and generating answer..."):

            try:
                # =========================
                # EXTRACT VIDEO ID
                # =========================
                match = re.search(r"v=([^&]+)", video_url)
                video_id = match.group(1) if match else video_url

                # =========================
                # GET TRANSCRIPT
                # =========================
                ytt_api = YouTubeTranscriptApi()
                transcript_list = ytt_api.fetch(video_id, languages=["en"])
                transcript = " ".join(chunk.text for chunk in transcript_list)

                # =========================
                # SPLIT TEXT
                # =========================
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.create_documents([transcript])

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
Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't know".

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

            except Exception as e:
                st.error(f"Error: {str(e)}")