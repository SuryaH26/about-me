"""
Resume Q&A Chatbot — powered by LangChain + OpenAI
Run with:  streamlit run app.py
"""

import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ── Config ─────────────────────────────────────────────────────────────────────
# Path to your resume PDF — relative to this file, or use an absolute path.
RESUME_PATH = Path(__file__).parent / "resume.pdf"

# FAISS index is persisted here so it's not rebuilt on every restart.
FAISS_INDEX_PATH = Path(__file__).parent / "faiss_index"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Chatbot",
    page_icon="📄",
    layout="centered",
)

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; padding: 8px 12px; }
    .main-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .sub-title  { color: #888; font-size: 0.95rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    google_api_key = "AIzaSyAZX9Cnn0GvWtYBXjiHJy1y580B_SeG7uQ"

    st.divider()
    st.markdown("**Model settings**")
    model_name = st.selectbox(
        "Model", ["gemini-2.5-flash"], index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    rebuild = st.button(
        "🔄 Rebuild Index", use_container_width=True,
        help="Force re-embed the resume (e.g. after updating the PDF).",
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if "chain" in st.session_state:
            del st.session_state["chain"]
        st.rerun()

    st.divider()
    st.caption(f"Resume: `{RESUME_PATH.name}`")
    st.caption("Built with LangChain · FAISS · Gemini")


# ── Helper: build / load FAISS index ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_chain(api_key: str, model: str, temp: float, _force_rebuild: bool = False):
    """
    Load resume → chunk → embed → FAISS.
    Persists the index to disk so subsequent starts skip the embedding step.
    Pass _force_rebuild=True to re-embed (e.g. after updating the PDF).
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    # ── Load or build vector store ───────────────────────────────────────────
    if FAISS_INDEX_PATH.exists() and not _force_rebuild:
        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        if not RESUME_PATH.exists():
            st.error(
                f"Resume PDF not found at: `{RESUME_PATH}`\n\n"
                "Place your PDF in the project folder and rename it `resume.pdf`, "
                "or update `RESUME_PATH` at the top of `app.py`."
            )
            st.stop()

        loader    = PyPDFLoader(str(RESUME_PATH))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(FAISS_INDEX_PATH))   # persist for next run

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10},
    )

    # ── Prompt ───────────────────────────────────────────────────────────────
    qa_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""You are a helpful assistant with deep knowledge of the candidate's
resume. Answer questions accurately using only the resume content provided.
If the information is not in the resume, say so clearly — do not fabricate.

Resume context:
{context}

Chat history:
{chat_history}

Question: {question}

Answer (be concise and professional):""",
    )

    # ── Memory + Chain ────────────────────────────────────────────────────────
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temp,
        google_api_key=api_key,
        convert_system_message_to_human=True,   # required for older gemini-pro
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )
    return chain


# ── Session state init ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Main UI ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📄 Resume Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ask anything about the resume — powered by Gemini.</p>', unsafe_allow_html=True)

if not google_api_key:
    st.info("👈 Enter your Google API key in the sidebar to get started.")
    st.stop()

# Build (or reload) chain
with st.spinner("📚 Loading resume index…"):
    chain = build_chain(
        api_key=google_api_key,
        model=model_name,
        temp=temperature,
        _force_rebuild=rebuild,
    )

index_status = "rebuilt ✅" if rebuild else "loaded from cache ✅"
st.success(f"Resume index {index_status}", icon="📄")

# ── Suggested starter questions ──────────────────────────────────────────────
STARTERS = [
    "What is the candidate's total years of experience?",
    "What are the key technical skills?",
    "Summarise the most recent role.",
    "What certifications or education does the candidate have?",
    "What AI/ML projects has the candidate worked on?",
]

if not st.session_state.messages:
    st.markdown("**💡 Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(STARTERS):
        if cols[i % 2].button(q, key=f"starter_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# ── Render chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about the resume…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = chain.invoke({"question": prompt})
            answer = result.get("answer", "Sorry, I couldn't find an answer.")
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})