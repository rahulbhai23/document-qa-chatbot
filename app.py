"""DocuMind - Document Q&A assistant, built with Streamlit components only.

Multi-file upload, real-time Q&A, source citations, conversation memory.
The page renders immediately; the AI models load only when first needed.
"""
import os
import tempfile

import streamlit as st
from rag_pipeline import DocumentQAEngine, format_citation_label

st.set_page_config(
    page_title="DocuMind | Document Q&A", page_icon="📄", layout="wide",
    initial_sidebar_state="expanded")

APP_NAME = "DocuMind"
SUGGESTIONS = [
    "Summarize the key points of these documents",
    "What are the main conclusions or recommendations?",
    "List any important dates, numbers or deadlines",
]


@st.cache_resource(show_spinner=False)
def get_engine():
    # Imported lazily so the page renders before the heavy models load.
    return DocumentQAEngine()


def load_engine():
    with st.spinner("Loading AI models... the first run downloads them and can take a few minutes."):
        return get_engine()


def make_sources(docs):
    """Convert retrieved LangChain documents into plain dicts for display."""
    return [{"label": format_citation_label(d), "text": d.page_content} for d in docs]


def render_sources(sources):
    with st.expander(f"📎 {len(sources)} sources used"):
        for i, s in enumerate(sources, 1):
            with st.container(border=True):
                st.markdown(f"**[{i}] {s['label']}**")
                text = s["text"][:400] + ("…" if len(s["text"]) > 400 else "")
                st.caption(text)


def transcript():
    lines = [f"{APP_NAME} conversation", ""]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else APP_NAME
        lines += [f"{who}: {m['content']}", ""]
        for i, s in enumerate(m.get("sources") or [], 1):
            lines.append(f"   [{i}] {s['label']}")
        lines.append("")
    return "\n".join(lines)


# ---------------- state ----------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("indexed", [])
st.session_state.setdefault("chunks", 0)

# ---------------- sidebar ----------------
with st.sidebar:
    st.title(f"📄 {APP_NAME}")
    st.caption("AI assistant for your documents")
    st.divider()

    st.subheader("1 · Upload documents")
    uploads = st.file_uploader(
        "PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
    if st.button("⚡ Process documents", type="primary", use_container_width=True,
                 disabled=not uploads):
        new_files = [u for u in uploads if u.name not in st.session_state.indexed]
        if not new_files:
            st.info("These files are already processed.")
        else:
            engine = load_engine()
            with st.spinner("Reading, chunking and indexing..."):
                tmp_files = []
                try:
                    for u in new_files:
                        suffix = os.path.splitext(u.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(u.getbuffer())
                            tmp_files.append((tmp.name, u.name))
                    n_chunks = engine.add_documents(tmp_files)
                    st.session_state.indexed += [u.name for u in new_files]
                    st.session_state.chunks += n_chunks
                    st.success(f"Added {len(new_files)} file(s).")
                except Exception as e:
                    st.error(f"Could not process files: {e}")
                finally:
                    for path, _ in tmp_files:
                        try:
                            os.unlink(path)
                        except OSError:
                            pass

    if st.session_state.indexed:
        st.markdown("**Indexed files**")
        for name in st.session_state.indexed:
            st.markdown(f"📎 {name}")

    st.divider()
    st.subheader("2 · Ask questions")
    st.caption("Answers come only from your files and cite their sources.")

    if st.session_state.messages:
        st.download_button(
            "⬇ Download conversation", transcript(), file_name="conversation.txt",
            use_container_width=True)
        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ---------------- header ----------------
st.title(APP_NAME)
st.caption("Ask questions about your PDF and Word files and get answers backed by cited sources.")

n_questions = sum(1 for m in st.session_state.messages if m["role"] == "user")
c1, c2, c3 = st.columns(3)
for col, value, label in [
    (c1, len(st.session_state.indexed), "Documents"),
    (c2, st.session_state.chunks, "Passages indexed"),
    (c3, n_questions, "Questions asked"),
]:
    with col.container(border=True):
        st.metric(label, value)

# ---------------- empty states ----------------
if not st.session_state.indexed:
    with st.container(border=True):
        st.subheader("Get started in three steps")
        st.markdown(
            "1. Upload one or more **PDF** or **DOCX** files in the sidebar.\n"
            "2. Click **Process documents** to index them.\n"
            "3. Ask a question in the chat box and check the sources behind each answer.")
elif not st.session_state.messages:
    st.subheader("Try a question")
    cols = st.columns(len(SUGGESTIONS))
    for col, q in zip(cols, SUGGESTIONS):
        if col.button(q, use_container_width=True, key=f"sugg_{q}"):
            st.session_state.pending = q
            st.rerun()

# ---------------- conversation ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "📄"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

question = st.chat_input("Ask a question about your documents...")
if not question and "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    if not st.session_state.indexed:
        st.warning("Upload and process at least one document first (see the sidebar).")
    else:
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="📄"):
            engine = load_engine()
            with st.spinner("Searching your documents..."):
                history = [(m["role"], m["content"]) for m in st.session_state.messages]
                try:
                    answer, docs = engine.generate_answer(question, chat_history=history)
                except Exception as e:
                    answer, docs = f"Something went wrong while answering: {e}", []
        sources = make_sources(docs) if docs else []
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources})
        st.rerun()