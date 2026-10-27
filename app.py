import json
import html
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from backend.btw_handler import handle_btw
from backend.llm import llm as _rename_llm
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, list_papers

st.set_page_config(page_title="Paperwise | Research Assistant", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #18212f;
        --muted: #667085;
        --line: #e8eaf0;
        --canvas: #f7f8fb;
        --violet: #6558d3;
        --violet-dark: #5145bd;
    }
    html, body, [class*="css"] {
        font-family: "Segoe UI", sans-serif;
        color: var(--ink);
    }
    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(227, 223, 255, .50), transparent 34rem),
            var(--canvas);
    }
    [data-testid="stHeader"] {
        background: rgba(247, 248, 251, .82);
        backdrop-filter: blur(14px);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding: 2.1rem 3rem 7rem;
    }
    [data-testid="stSidebar"] {
        background: #fff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.1rem 2rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
        font-size: .9rem;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        border: 1px dashed #c9c7e8;
        border-radius: 14px;
        background: #fafaff;
    }
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] input {
        border-radius: 12px;
        border-color: var(--line);
        background: #fbfbfd;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 0 0 1.25rem;
    }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        border-radius: 14px;
        color: white;
        font-size: 22px;
        font-weight: 800;
        background: linear-gradient(145deg, #7668e7, #5145bd);
        box-shadow: 0 8px 20px rgba(93, 79, 205, .24);
    }
    .brand-name {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -.04em;
    }
    .brand-caption {
        margin-top: 1px;
        color: var(--muted);
        font-size: .76rem;
    }
    .hero {
        overflow: hidden;
        padding: 2.6rem 2.8rem;
        border: 1px solid #e1dff0;
        border-radius: 25px;
        background:
            radial-gradient(circle at 90% 18%, rgba(194, 187, 255, .48), transparent 18rem),
            linear-gradient(120deg, #fff 0%, #fbfaff 62%, #f1efff 100%);
        box-shadow: 0 16px 44px rgba(30, 37, 68, .055);
    }
    .hero-kicker {
        display: inline-flex;
        gap: 8px;
        margin-bottom: 1.15rem;
        padding: 6px 11px;
        border: 1px solid #e9e6ff;
        border-radius: 99px;
        background: rgba(255,255,255,.8);
        color: var(--violet-dark);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .045em;
        text-transform: uppercase;
    }
    .hero h1 {
        max-width: 700px;
        margin: 0;
        font-size: clamp(2.25rem, 5vw, 3.65rem);
        font-weight: 800;
        letter-spacing: -.065em;
        line-height: 1.08;
    }
    .hero h1 span { color: var(--violet); }
    .hero-copy {
        max-width: 650px;
        margin: 1rem 0 0;
        color: #606b7b;
        font-size: 1.02rem;
        line-height: 1.75;
    }
    .hero-foot {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 1.6rem;
    }
    .hero-chip {
        padding: 7px 11px;
        border: 1px solid #e9eaf0;
        border-radius: 10px;
        background: rgba(255,255,255,.75);
        color: #526071;
        font-size: .8rem;
        font-weight: 600;
    }
    .section-heading {
        margin: 2rem 0 .25rem;
        font-size: 1.2rem;
        font-weight: 750;
        letter-spacing: -.035em;
    }
    .section-subtitle {
        margin-bottom: 1rem;
        color: var(--muted);
        font-size: .92rem;
    }
    .feature-card {
        height: 100%;
        min-height: 168px;
        padding: 1.2rem 1.25rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(255,255,255,.92);
        box-shadow: 0 8px 24px rgba(31, 41, 55, .035);
    }
    .feature-icon {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        margin-bottom: .9rem;
        border-radius: 12px;
        background: #f0efff;
        font-size: 19px;
    }
    .feature-card h3 {
        margin: 0 0 .4rem;
        font-size: 1rem;
        font-weight: 750;
        letter-spacing: -.025em;
    }
    .feature-card p {
        margin: 0;
        color: var(--muted);
        font-size: .86rem;
        line-height: 1.6;
    }
    .session-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: .15rem 0 1.4rem;
        padding: .8rem 1rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: rgba(255,255,255,.8);
    }
    .session-label {
        color: var(--muted);
        font-size: .82rem;
    }
    .session-name {
        overflow: hidden;
        max-width: 65%;
        font-weight: 700;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    [data-testid="stChatMessage"] {
        padding: 1rem 1.15rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(255,255,255,.94);
        box-shadow: 0 6px 20px rgba(31, 41, 55, .025);
    }
    [data-testid="stChatInput"] {
        border: 1px solid #deddeb;
        border-radius: 18px;
        background: #fff;
        box-shadow: 0 10px 34px rgba(31, 41, 55, .10);
    }
    .stButton > button {
        border-radius: 12px;
        font-weight: 650;
        transition: transform .16s ease, box-shadow .16s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(81, 69, 189, .15);
    }
    [data-testid="stExpander"] {
        border-color: var(--line);
        border-radius: 14px;
        background: rgba(255,255,255,.7);
    }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.25rem 1rem 6rem;
        }
        .hero {
            padding: 1.65rem 1.35rem;
            border-radius: 20px;
        }
        .hero-copy { font-size: .94rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
                chats.append({"role": "assistant", "content": content, "turn": turn, "graph_state": {}})
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


graph = get_graph()

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">P</div>
          <div>
            <div class="brand-name">Paperwise</div>
            <div class="brand-caption">Your research workspace</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("＋  New research chat", use_container_width=True, type="primary"):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()
    st.markdown("### Recent chats")

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()

    st.divider()
    st.markdown("### Your sources")
    st.caption("Add papers and webpages to ground your answers.")

    # ── Section 1: File upload ─────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
    )
    if st.button("Add to library", use_container_width=True, key="btn_add_files"):
        if uploaded_files:
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
            with st.spinner("Processing files…"):
                for f in uploaded_files:
                    if f.name in st.session_state[processed_key]:
                        st.info(f"Already loaded: {f.name}")
                        continue
                    suffix = Path(f.name).suffix
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        docs = load_document(tmp_path)
                        for doc in docs:
                            doc.metadata["title"] = Path(f.name).stem
                        add_paper(docs, active_sid)
                        st.session_state[processed_key].add(f.name)
                        st.success(f"Added: {f.name}")
                    except Exception as e:
                        st.error(f"Failed: {f.name} — {e}")
                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)
            st.rerun()
        else:
            st.warning("No files selected.")

    # ── Section 2: Web URL loader ──────────────────────────────────────────────
    st.markdown("**Add a webpage**")
    url_input = st.text_area(
        "URLs (one per line)",
        key=f"url_area_{active_sid}",
        height=80,
        label_visibility="collapsed",
        placeholder="https://example.com/paper",
    )
    if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if urls:
            with st.spinner("Loading web pages…"):
                for url in urls:
                    try:
                        docs = load_webpage(url)
                        add_paper(docs, active_sid)
                        st.success(f"Loaded: {url[:60]}")
                    except Exception as e:
                        st.error(f"Failed: {url[:60]} — {e}")
            st.rerun()
        else:
            st.warning("Enter at least one URL.")

    # ── Section 3: ArXiv loader ────────────────────────────────────────────────
    st.markdown("**Find an arXiv paper**")
    arxiv_title = st.text_input(
        "Paper title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="1706.03762  or  Attention Is All You Need",
    )
    if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
        if arxiv_title.strip():
            with st.spinner("Loading from ArXiv…"):
                try:
                    docs = load_arxiv(arxiv_title.strip())
                    add_paper(docs, active_sid)
                    loaded_title = docs[0].metadata.get("title") if docs else arxiv_title.strip()
                    st.success(f"Loaded: {loaded_title}")
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()
        else:
            st.warning("Enter a paper title or ArXiv ID.")

    # ── Loaded Documents list ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### In this library")
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None
    if doc_titles is None:
        st.caption("Could not load document list — try refreshing.")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(f"- {title}")
    else:
        st.caption("No documents loaded yet.")

# ── Workspace header ───────────────────────────────────────────────────────────
active_session = st.session_state.sessions_meta.get(active_sid, {})
active_session_name = html.escape(active_session.get("name", "Research chat"))
chat_history = st.session_state.chats.get(active_sid, [])
document_count = len(doc_titles) if isinstance(doc_titles, list) else 0
model_name = html.escape(getattr(_rename_llm, "model", "NVIDIA AI"))

st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker"><span>✦</span> AI-powered research desk</div>
      <h1>Make sense of your<br><span>research papers.</span></h1>
      <p class="hero-copy">
        Bring your papers together, ask better questions, and get answers grounded
        in the sources you trust.
      </p>
      <div class="hero-foot">
        <span class="hero-chip">⌕ &nbsp;Paper-grounded answers</span>
        <span class="hero-chip">↗ &nbsp;Current web research</span>
        <span class="hero-chip">✓ &nbsp;Claim verification</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="session-bar">
      <span class="session-label">CURRENT CHAT</span>
      <span class="session-name">{active_session_name}</span>
      <span class="session-label">{document_count} source{"s" if document_count != 1 else ""} · {model_name}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if not chat_history:
    st.markdown(
        '<div class="section-heading">A little help goes a long way</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Choose a starting point. Add a source in the sidebar to get grounded answers.</div>',
        unsafe_allow_html=True,
    )
    feature_columns = st.columns(3, gap="medium")
    feature_cards = [
        (
            "⌕",
            "Ask your papers",
            "Compare methods, pull out key findings, or get a clear summary with answers based on your library.",
        ),
        (
            "↗",
            "Explore the latest",
            "Use <code>/btw</code> before a question to get a separate answer that can search the live web.",
        ),
        (
            "✓",
            "Check a claim",
            "Ask whether a paper's finding has been challenged or updated by more recent work.",
        ),
    ]
    for column, (icon, title, description) in zip(feature_columns, feature_cards):
        with column:
            st.markdown(
                f"""
                <article class="feature-card">
                  <div class="feature-icon">{icon}</div>
                  <h3>{title}</h3>
                  <p>{description}</p>
                </article>
                """,
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="section-subtitle" style="margin-top:1.2rem">Try asking: “What are the main findings?” · “Compare the authors’ methods” · “What should I read next?”</div>',
        unsafe_allow_html=True,
    )

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander(f"📊 Graph state · turn {msg['turn']}", expanded=False):
                st.json(msg["graph_state"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…"):
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side channel — not saved to session history.")

        with st.chat_message("assistant"):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
            else:
                placeholder = st.empty()
                response_text = ""
                for chunk in handle_btw(query):
                    response_text += chunk
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)
            st.caption("Side channel — not saved to session history.")

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1
        current_turn = st.session_state.turns[active_sid]

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": active_sid}}

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_text = ""

            for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                if (
                    metadata.get("langgraph_node") == "generate_answer"
                    and hasattr(chunk, "content")
                    and chunk.content
                ):
                    response_text += chunk.content
                    placeholder.markdown(response_text + "▌")

            if not response_text:
                final_values = graph.get_state(config).values
                response_text = final_values.get("answer") or "No response generated."

            placeholder.markdown(response_text)

            final_values = graph.get_state(config).values
            state_snapshot = _serialize_state(final_values)

            with st.expander(f"📊 Graph state · turn {current_turn}", expanded=False):
                st.json(state_snapshot)

        st.session_state.chats[active_sid].append(
            {
                "role": "assistant",
                "content": response_text,
                "graph_state": state_snapshot,
                "turn": current_turn,
            }
        )

        if is_first_message:
            st.rerun()
