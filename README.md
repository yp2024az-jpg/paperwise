# PAPERWISE

Paperwise is an autonomous research paper assistant powered by a LangGraph agentic workflow. It enables users to upload academic papers, execute multi-step reasoning loops to query document contents, cross-verify research claims against real-time web results using Tavily, and handle general technical inquiries via an integrated `/btw` chat routing mechanism.

The application features a Streamlit frontend coupled with an agentic backend orchestration layer. It leverages NVIDIA models for chat inference and embeddings, Qdrant for vector storage, Tavily for live web search, and DeepEval for programmatic evaluation scripts.

I haven't been able to get this project uploaded to GitHub yet.

## Key Agentic Features & Architecture

- **Multi-Agent & Tool-Calling Workflows:** Built using **LangGraph**, orchestrating autonomous agent loops that decide whether to query the local vector store, search the live web, or fall back to general knowledge.
- **Dynamic Document Ingestion:** Upload PDF, TXT, and Markdown files, or load papers dynamically via ArXiv ID/title and index web pages.
- **Claim Verification Loop:** Cross-references extracted research claims against recent web publications to ensure assertions have not been superseded by newer findings.
- **Isolated Side-Channel Chat (`/btw`):** Utilizes custom routing logic to handle general knowledge or standalone web search queries separately from the active document context.
- **Automated RAG Evaluation:** Integrated testing suite powered by DeepEval running against golden datasets to track strict execution accuracy and telemetry.
  
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
