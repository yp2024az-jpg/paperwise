# ODIN

I built Paperwise as a small research paper assistant. You can upload papers and then ask questions about the content of those papers, check claims against the latest web results and use a separate /btw chat for questions that aren't specifically about the papers you've uploaded.

The app is a Streamlit frontend with a LangGraph workflow. It uses NVIDIA models for chat and embeddings, Qdrant for vector store, Tavily for web search and DeepEval for the eval script.

I haven't been able to get this project uploaded to GitHub yet.

## What it can do

- Upload PDF, TXT and Markdown files.
- Load papers by ArXiv ID or title, and index web pages.
- Pose questions using the documents uploaded to the current chat session.
- Check that a research claim hasn't been superseded by more recent work.
- Use `/btw <question>` for a standalone general knowledge or web search response.
- Check the answers against the OpenClaw report and the saved questions.

## Setup

Use Python 3.10 or later. From the project root, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create or update backend/.env with the keys and endpoints listed below. Keep any real credentials private and don't commit them.

```dotenv
NVIDIA_API_KEY=your_nvidia_api_key
nvidia_api=your_nvidia_api_key
api_key=your_qdrant_api_key
Cluster_Endpoint=your_qdrant_cluster_url
tavily=your_tavily_api_key
```

The chat model reads 'NVIDIA_API_KEY', the embedding client currently reads 'nvidia_api'. Both are NVIDIA credentials. Qdrant and Tavily use the names as shown above.

## Run the app

From the project root:

```powershell
python -m streamlit run app.py
```

Streamlit will print out the local URL after it's started up. Add your documents to the sidebar prior to asking questions about them.

## Run the evaluation

evaluate.py uses Openclaw_Research_Report.pdf and the questions in goldens.json. Run it from the project root:

```powershell
python evaluate.py
```

The evaluation makes a lot of NVIDIA API calls for the RAG answers and for the metric scoring. **You need to use a paid NVIDIA API key with sufficient quota. Don't run it on the free tier as you'll very quickly hit the rate limits and have cases fail as a result.**

To run the questions in chunks use zero based indices and an exclusive end index. For the ten saved questions:

```powershell
python evaluate.py --start-index 0 --end-index 5
python evaluate.py --start-index 5 --end-index 10
```

Results are saved in eval_results.json, errors are logged there so that in the event of a failed run you can just re-run and it'll pick up where it left off.