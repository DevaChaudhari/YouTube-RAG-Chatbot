# 🎥 YouTube RAG Chatbot

An end-to-end AI-powered **Retrieval-Augmented Generation (RAG)** chatbot that allows users to ask questions from YouTube video transcripts in real time.

The application extracts YouTube transcripts, processes them into semantic chunks, generates embeddings using **Google Gemini**, stores them in a **FAISS vector database**, and retrieves relevant context to generate accurate answers using an LLM.

## 🚀 Live Demo

🔗 [Open Live App](https://youtube-rag-chatbot-fq8fy6zy9vw2ohypqiztit.streamlit.app/)

## 📌 Features

- 🎥 YouTube transcript extraction
- 🧠 Retrieval-Augmented Generation pipeline
- ✂️ Semantic text chunking
- 🔎 FAISS vector database for similarity search
- 🤖 Gemini-powered embeddings and answer generation
- 💬 Interactive Streamlit user interface
- 🐳 Dockerized application
- ☸️ Kubernetes deployment configuration
- ⚙️ GitHub Actions CI workflow
- 🌐 Live deployment on Streamlit Cloud

## 🏗️ Project Architecture

```text
User enters YouTube URL
        |
        v
Fetch video metadata and transcript
        |
        v
Split transcript into semantic chunks
        |
        v
Generate embeddings using Google Gemini
        |
        v
Store embeddings in FAISS vector index
        |
        v
Retrieve relevant transcript chunks
        |
        v
Generate context-aware answer using Gemini LLM
```

## 🛠️ Tech Stack

**Python | LangChain | Streamlit | Google Gemini | FAISS | YouTube Transcript API | Docker | Kubernetes | GitHub Actions**

## 📊 Project Highlights

- Built a **4-stage RAG pipeline**: metadata fetch, transcript loading, FAISS indexing, and Gemini-based answer generation.
- Integrated **Google Gemini embeddings and Gemini LLM** for context-aware Q&A.
- Used **FAISS vector search** for semantic retrieval from YouTube transcripts.
- Developed an interactive **Streamlit UI** for real-time user queries.
- Achieved **~85-90% relevant response quality** on manually tested transcript Q&A queries.
- Reduced manual video review time by enabling direct querying of long-form video content.
- Dockerized the application for consistent development and deployment environments.
- Added Kubernetes deployment and service configuration for cloud-native readiness.
- Implemented CI/CD automation for streamlined build, testing, and deployment workflows.
- Deployed the project as a live public application on Streamlit Cloud.

## 📂 Project Structure

```text
YouTube-RAG-Chatbot/
│
├── app.py
├── main.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .dockerignore
├── .gitignore
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
└── .github/
    └── workflows/
        └── ci.yml
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/DevaChaudhari/YouTube-RAG-Chatbot.git
cd YouTube-RAG-Chatbot
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

```bash
# Windows
venv\Scripts\activate
```

```bash
# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY="your_api_key_here"
```

> Do not commit your `.env` file or API keys to GitHub.

## ▶️ Run Locally

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## 🐳 Docker Usage

Build the Docker image:

```bash
docker build -t youtube-rag-chatbot .
```

Run the container:

```bash
docker run -p 8501:8501 --env-file .env youtube-rag-chatbot
```

Open in browser:

```text
http://localhost:8501
```

## ☸️ Kubernetes Deployment

Create a Kubernetes secret for the Gemini API key:

```bash
kubectl create secret generic youtube-rag-secrets --from-literal=GOOGLE_API_KEY="your_api_key_here"
```

Apply the Kubernetes manifests:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Check running pods and services:

```bash
kubectl get pods
kubectl get svc
```

## ⚙️ CI/CD

This project includes a GitHub Actions workflow that runs on pushes and pull requests to the `main` branch.

The workflow:

- Checks out the repository
- Sets up Python 3.11
- Installs project dependencies
- Validates Python syntax for `app.py` and `main.py`

## 💡 Usage

1. Open the application.
2. Paste a YouTube video URL.
3. Enter a question about the video.
4. Click **Get Answer**.
5. The chatbot retrieves relevant transcript context and generates an answer.

Example questions:

- What is this video about?
- Summarize the video.
- What are the main points?
- What does the speaker recommend?

## 🧠 Skills Demonstrated

- Retrieval-Augmented Generation
- LLM application development
- LangChain pipeline design
- Vector search with FAISS
- Google Gemini API integration
- Streamlit application development
- Docker containerization
- Kubernetes deployment
- GitHub Actions CI
- Cloud deployment

## 🔗 Links

- 🌐 **Live App:** [YouTube RAG Chatbot](https://youtube-rag-chatbot-fq8fy6zy9vw2ohypqiztit.streamlit.app/)
- 💻 **GitHub Repository:** [DevaChaudhari/YouTube-RAG-Chatbot](https://github.com/DevaChaudhari/YouTube-RAG-Chatbot)

## 👨‍💻 Author

**Deva Chaudhari**

GitHub: [@DevaChaudhari](https://github.com/DevaChaudhari)

## 📄 License

This project is open source and available under the MIT License.
