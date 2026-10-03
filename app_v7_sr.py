import os
import sys
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

# Corporate proxy for external LLM calls when configured in .env.
for _name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
    if os.getenv(_name):
        os.environ[_name] = os.getenv(_name)

import streamlit as st

# Chroma requires a newer SQLite than the Oracle Linux 8 system SQLite.
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Exadata Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Exadata Chatbot")

KNOWLEDGE_BASES = {
    "Exadata On-Prem": {
        "db_folder": APP_DIR / "vectordb_exadata_onprem",
        "collection_name": "exadata_docs",
        "label": "Oracle Exadata Database Machine",
        "type": "public",
        "source_urls": [
            "https://docs.oracle.com/en/engineered-systems/exadata-database-machine/dbmin/toc.htm",
            "https://docs.oracle.com/en/engineered-systems/exadata-database-machine/sagug/toc.htm",
            "https://docs.oracle.com/en/engineered-systems/exadata-database-machine/dbmso/toc.htm",
            "https://docs.oracle.com/en/engineered-systems/exadata-database-machine/dbmsq/toc.htm",
            "https://docs.oracle.com/en/engineered-systems/exadata-database-machine/dbmmn/toc.htm",
        ],
    },
    "Exascale": {
        "db_folder": APP_DIR / "vectordb_exascale",
        "collection_name": "exadata_docs",
        "label": "Oracle Exadata Exascale",
        "type": "public",
        "source_urls": [
            "https://docs.oracle.com/en/engineered-systems/exadata-database-machine/exscl/toc.htm",
        ],
    },
    "ExaCC": {
        "db_folder": APP_DIR / "vectordb_exacc",
        "collection_name": "exadata_docs",
        "label": "Oracle Exadata Database Service on Cloud@Customer",
        "type": "public",
        "source_urls": [
            "https://docs.oracle.com/en/engineered-systems/exadata-cloud-at-customer/ecccm/toc.htm",
        ],
    },
    "ExaCS": {
        "db_folder": APP_DIR / "vectordb_exacs",
        "collection_name": "exadata_docs",
        "label": "Oracle Exadata Database Service on Dedicated Infrastructure",
        "type": "public",
        "source_urls": [
            "https://docs.oracle.com/en/engineered-systems/exadata-cloud-service/ecscm/toc.htm",
        ],
    },
    "Oracle Knowledge": {
        "db_folder": APP_DIR / "vectordb_oracle_knowledge",
        "collection_name": "oracle_knowledge",
        "label": "Oracle Knowledge / KM",
        "type": "internal",
        "source_urls": [],
    },
    "Service Requests": {
        "db_folder": APP_DIR / "vectordb_sr",
        "collection_name": "service_requests",
        "label": "Historical Oracle Service Requests",
        "type": "internal_sr",
        "source_urls": [],
    },
}

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "selected_kb" not in st.session_state:
    st.session_state.selected_kb = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configuration")

    selected_kb = st.selectbox(
        "Select knowledge source",
        options=list(KNOWLEDGE_BASES.keys()),
        help="Search only the selected knowledge base.",
    )

    if st.session_state.selected_kb != selected_kb:
        st.session_state.selected_kb = selected_kb
        st.session_state.messages = []

    kb = KNOWLEDGE_BASES[selected_kb]

    st.divider()
    st.subheader("LLM Configuration")

    llm_provider = st.selectbox(
        "LLM provider",
        options=[
            "No LLM - Documentation Search",
            "OpenAI-compatible",
            "Google Gemini",
        ],
    )

    if llm_provider == "No LLM - Documentation Search":
        llm_model = ""
        llm_base_url = ""
        llm_api_key = ""
        st.info(
            "🔎 No LLM mode: retrieve relevant documents and show source links "
            "without generating an LLM answer."
        )

    elif llm_provider == "OpenAI-compatible":
        llm_model = st.text_input(
            "Model name",
            value=os.getenv("LLM_MODEL", ""),
            placeholder="Example: gpt-4o-mini",
        )
        llm_base_url = st.text_input(
            "Base URL",
            value=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        )
        llm_api_key = st.text_input(
            "Your API key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Kept only in this Streamlit session; not written to disk.",
        )
        st.session_state.openai_api_key = llm_api_key
        if not llm_api_key:
            st.caption("Enter your own OpenAI-compatible API key.")

    else:
        llm_model = st.text_input(
            "Gemini model",
            value=os.getenv("GEMINI_MODEL", ""),
            placeholder="Example: gemini-3.5-flash",
        )
        llm_base_url = ""
        llm_api_key = st.text_input(
            "Your Gemini API key",
            value=st.session_state.gemini_api_key,
            type="password",
            help="Kept only in this Streamlit session; not written to disk.",
        )
        st.session_state.gemini_api_key = llm_api_key
        if not llm_api_key:
            st.caption("Enter your own Gemini API key.")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        disabled=(llm_provider == "Google Gemini"),
        help="Used for providers that support configurable sampling temperature.",
    )

    st.divider()
    st.caption(f"Knowledge source: {kb['label']}")
    st.caption(f"Chroma collection: `{kb['collection_name']}`")

    if kb["type"] == "public":
        st.caption("Source type: Public Oracle documentation")
        if kb["source_urls"]:
            st.caption("Documentation roots used during ingestion:")
            for source_url in kb["source_urls"]:
                st.markdown(f"- [{source_url}]({source_url})")
    elif kb["type"] == "internal_sr":
        st.caption("Source type: Historical Oracle Service Requests")
        st.caption("Retrieved SRs include an Open SR link.")
    else:
        st.caption("Source type: Oracle internal Knowledge / KM")
        st.caption("Retrieved KM articles include an Open KB Article link.")

if kb["type"] == "public":
    st.markdown(
        """
        <div style="background-color:#fff1f0;border:1px solid #ff4d4f;"
        "border-left:5px solid #d32f2f;padding:12px 16px;border-radius:6px;"
        "margin-bottom:20px;color:#b71c1c;font-weight:600;">
        ⚠️ <strong>Public Documentation:</strong>
        This knowledge source contains publicly available Oracle documentation only.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div style="background-color:#fff7e6;border:1px solid #faad14;"
        "border-left:5px solid #d48806;padding:12px 16px;border-radius:6px;"
        "margin-bottom:20px;color:#7a4d00;font-weight:600;">
        🔐 <strong>Internal Knowledge:</strong>
        This source contains Oracle internal Knowledge/KM content. Use only within
        the authorized internal team and do not share responses outside approved users.
        </div>
        """,
        unsafe_allow_html=True,
    )

if llm_provider != "No LLM - Documentation Search" and not llm_model.strip():
    st.info("Please enter an LLM model name in the sidebar to begin.")
    st.stop()

if llm_provider in {"OpenAI-compatible", "Google Gemini"} and not llm_api_key.strip():
    st.info("Please enter your API key in the sidebar to begin.")
    st.stop()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)

@st.cache_resource
def load_embedding(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

try:
    embedding = load_embedding(EMBEDDING_MODEL)
except Exception as exc:
    st.error(f"Unable to load the embedding model `{EMBEDDING_MODEL}`. Error: {exc}")
    st.stop()

db_folder = Path(kb["db_folder"])
if not db_folder.exists():
    st.error(
        f"Vector database directory `{db_folder}` was not found. "
        "Run the ingestion notebook for this knowledge source first."
    )
    st.stop()

@st.cache_resource
def load_vector_db(folder, collection_name):
    return Chroma(
        persist_directory=str(folder),
        embedding_function=embedding,
        collection_name=collection_name,
    )

try:
    db = load_vector_db(
    db_folder,
    kb["collection_name"],
)
except Exception as exc:
    st.error(f"Unable to load Chroma database `{db_folder}`. Error: {exc}")
    st.stop()

retriever = db.as_retriever(search_kwargs={"k": 60})

def rank_km_documents(query, docs, max_articles=5):
    """
    Group Oracle Knowledge chunks by KM article and rank articles using
    query/title/content matches. The goal is to prevent many chunks from
    the same version-specific article from crowding out general
    compatibility/support articles.
    """
    query_terms = {
        term.lower().strip(".,:;!?()[]{}\"'")
        for term in query.split()
        if len(term.strip(".,:;!?()[]{}\"'")) >= 3
    }

    intent_terms = {
        "minimum",
        "supported",
        "support",
        "versions",
        "version",
        "database",
        "db",
        "grid",
        "upgrade",
        "patch",
        "release",
        "compatibility",
        "requirement",
        "requirements",
    }

    grouped = {}

    for doc in docs:
        doc_id = (
            doc.metadata.get("doc_id")
            or doc.metadata.get("source")
            or doc.metadata.get("title")
            or str(id(doc))
        )

        title = (doc.metadata.get("title") or "").lower()
        content = (doc.page_content or "").lower()

        score = 0.0

        # Strong signal: query terms in the KM article title.
        score += 5.0 * sum(
            1 for term in query_terms if term in title
        )

        # Strong signal: support/version/upgrade intent in title.
        score += 3.0 * sum(
            1 for term in intent_terms if term in title
        )

        # Content match.
        score += 1.0 * sum(
            1 for term in query_terms if term in content
        )

        # Small boost when the article content contains several
        # support/version concepts.
        intent_hits = sum(
            1 for term in intent_terms if term in content
        )
        score += min(intent_hits, 8) * 0.25

        if doc_id not in grouped:
            grouped[doc_id] = {
                "score": score,
                "docs": [doc],
            }
        else:
            grouped[doc_id]["docs"].append(doc)
            grouped[doc_id]["score"] = max(
                grouped[doc_id]["score"],
                score,
            )

    ranked = sorted(
        grouped.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )

    return ranked[:max_articles]


def build_km_context(
    ranked_articles,
    max_chunks_per_article=4,
):
    """Select the best chunks from the best-ranked KM articles."""
    context_docs = []

    for _, article in ranked_articles:
        article_docs = sorted(
            article["docs"],
            key=lambda d: len(d.page_content or ""),
            reverse=True,
        )
        context_docs.extend(
            article_docs[:max_chunks_per_article]
        )

    return context_docs



def _query_terms(query):
    return {
        term.lower().strip(".,:;!?()[]{}\"'")
        for term in query.split()
        if len(term.strip(".,:;!?()[]{}\"'")) >= 3
    }


def rank_sr_documents(query, docs, max_articles=5):
    """Rank SR chunks at the SR level and group multiple chunks by SR."""
    import re

    query_terms = _query_terms(query)
    intent_terms = {
        "issue", "problem", "error", "failure", "patch", "upgrade",
        "bug", "ora", "rac", "asm", "acfs", "dataguard", "backup",
        "performance", "vulnerability", "swap", "network", "dns",
    }
    grouped = {}

    for doc in docs:
        sr_id = (
            doc.metadata.get("sr_id")
            or doc.metadata.get("source")
            or doc.metadata.get("title")
            or str(id(doc))
        )
        title = (doc.metadata.get("title") or "").lower()
        content = (doc.page_content or "").lower()
        score = 0.0

        score += 7.0 * sum(1 for term in query_terms if term in title)
        score += 1.5 * sum(1 for term in query_terms if term in content)
        score += 1.5 * sum(1 for term in intent_terms if term in content)

        for pattern in (r"\bora-?\d+\b", r"\bbug\s+\d+\b", r"\b4-\d{10}\b"):
            for token in re.findall(pattern, query.lower()):
                if token in title:
                    score += 12.0
                if token in content:
                    score += 6.0

        if sr_id not in grouped:
            grouped[sr_id] = {"score": score, "docs": [doc]}
        else:
            grouped[sr_id]["docs"].append(doc)
            grouped[sr_id]["score"] = max(grouped[sr_id]["score"], score)

    ranked = sorted(grouped.items(), key=lambda item: item[1]["score"], reverse=True)
    return ranked[:max_articles]


def build_sr_context(query, ranked_articles, max_chunks_per_article=4):
    """Select the most query-relevant chunks from each selected SR."""
    query_terms = _query_terms(query)
    context_docs = []

    for _, article in ranked_articles:
        scored = []
        for doc in article["docs"]:
            title = (doc.metadata.get("title") or "").lower()
            content = (doc.page_content or "").lower()
            score = (
                5.0 * sum(1 for term in query_terms if term in title)
                + 2.0 * sum(1 for term in query_terms if term in content)
            )
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        context_docs.extend(doc for _, doc in scored[:max_chunks_per_article])

    return context_docs

@st.cache_resource
def build_llm(provider, model, api_key, base_url, temp):
    if provider == "No LLM - Documentation Search":
        return None
    if provider == "Google Gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            timeout=30,
            max_retries=1,
        )
    kwargs = {
        "model": model,
        "temperature": temp,
        "timeout": 30,
        "max_retries": 1,
    }
    if api_key.strip():
        kwargs["api_key"] = api_key
    if base_url.strip():
        kwargs["base_url"] = base_url.rstrip("/")
    return ChatOpenAI(**kwargs)

try:
    llm = build_llm(llm_provider, llm_model.strip(), llm_api_key.strip(), llm_base_url.strip(), temperature)
except Exception as exc:
    st.error(f"Unable to initialize the selected LLM: {exc}")
    st.stop()

system_prompt = (
    f"You are an expert technical assistant specializing in {kb['label']}. "
    f"The selected knowledge source is {selected_kb}.\n\n"
    "Answer using only the retrieved documentation for the currently selected knowledge source. "
    "If the retrieved documentation does not contain the answer, say that you don't know. "
    "Do not introduce unsupported information or mix unrelated knowledge sources. "
    "Keep the answer concise and technical. When useful, mention the document title or identifier. "
    "For Service Requests, distinguish documented customer statements, Oracle support actions, "
    "findings, workarounds, and resolution status. Do not turn a historical SR statement into a general Oracle recommendation. "
    "Preserve important dates and qualifiers.\n\nContext:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input(f"Ask a technical question about {selected_kb}..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching {selected_kb} documentation..."):
            try:
                if llm_provider == "No LLM - Documentation Search":
                    retrieved_docs = retriever.invoke(user_query)

                    if not retrieved_docs:
                        full_output = "No relevant documentation was found."
                    else:
                        if selected_kb == "Oracle Knowledge":
                            ranked_articles = rank_km_documents(
                                user_query,
                                retrieved_docs,
                                max_articles=3,
                            )
                            # v7 build_km_context accepts ranked articles directly.
                            source_docs = build_km_context(
                                ranked_articles,
                                max_chunks_per_article=2,
                            )
                        elif selected_kb == "Service Requests":
                            ranked_articles = rank_sr_documents(
                                user_query,
                                retrieved_docs,
                                max_articles=5,
                            )
                            source_docs = build_sr_context(
                                user_query,
                                ranked_articles,
                                max_chunks_per_article=2,
                            )
                        else:
                            source_docs = retrieved_docs[:8]

                        grouped_display = {}
                        for doc in source_docs:
                            key = (
                                doc.metadata.get("doc_id")
                                or doc.metadata.get("sr_id")
                                or doc.metadata.get("source")
                                or doc.metadata.get("title")
                                or id(doc)
                            )
                            grouped_display.setdefault(key, []).append(doc)

                        parts = [f"### 🔎 Relevant {selected_kb} Documentation"]

                        for idx, docs_for_source in enumerate(grouped_display.values(), 1):
                            doc = docs_for_source[0]
                            source = doc.metadata.get("source", "Unknown source")
                            title = (
                                doc.metadata.get("title")
                                or doc.metadata.get("document_title")
                                or source.rsplit("/", 1)[-1]
                            )
                            identifier = (
                                doc.metadata.get("sr_id")
                                if selected_kb == "Service Requests"
                                else doc.metadata.get("doc_id")
                            )
                            content = "\n\n".join(
                                d.page_content.strip()
                                for d in docs_for_source[:2]
                                if d.page_content and d.page_content.strip()
                            )
                            if len(content) > 2400:
                                content = content[:2400] + "..."

                            label = (
                                f"SR {identifier}"
                                if selected_kb == "Service Requests" and identifier
                                else (f"KM ID: `{identifier}`" if identifier else "")
                            )
                            parts.append(
                                f"\n#### Result {idx}: {title}"
                                + (f" ({label})" if label else "")
                                + f"\n\n{content}\n\n"
                            )

                            if source and source != "Unknown source":
                                if selected_kb == "Service Requests" and identifier:
                                    parts.append(
                                        f"**Source:** SR `{identifier}`  \n"
                                        f"**Title:** {title}  \n"
                                        f"[🔗 Open SR]({source})\n\n---"
                                    )
                                elif identifier:
                                    parts.append(
                                        f"**Source:** KM Article `{identifier}`  \n"
                                        f"**Title:** {title}  \n"
                                        f"[🔗 Open KB Article]({source})\n\n---"
                                    )
                                else:
                                    parts.append(
                                        f"**Source:** [🔗 Open Source]({source})\n\n---"
                                    )

                        full_output = "\n".join(parts)

                else:
                    retrieved_docs = retriever.invoke(user_query)

                    if selected_kb == "Oracle Knowledge":
                        ranked_articles = rank_km_documents(
                            user_query,
                            retrieved_docs,
                            max_articles=3,
                        )
                        source_docs = build_km_context(
                            ranked_articles,
                            max_chunks_per_article=4,
                        )
                    elif selected_kb == "Service Requests":
                        ranked_articles = rank_sr_documents(
                            user_query,
                            retrieved_docs,
                            max_articles=5,
                        )
                        source_docs = build_sr_context(
                            user_query,
                            ranked_articles,
                            max_chunks_per_article=4,
                        )
                    else:
                        source_docs = retrieved_docs[:8]

                    question_answer_chain = create_stuff_documents_chain(
                        llm,
                        prompt,
                    )

                    assistant_response = question_answer_chain.invoke(
                        {
                            "input": user_query,
                            "context": source_docs,
                        }
                    )

                    citation_lines = []
                    seen_sources = set()
                    for doc in source_docs:
                        source = doc.metadata.get("source")
                        if not source or source in seen_sources:
                            continue
                        seen_sources.add(source)

                        title = (
                            doc.metadata.get("title")
                            or doc.metadata.get("document_title")
                            or source.rsplit("/", 1)[-1]
                            or source
                        )
                        identifier = (
                            doc.metadata.get("sr_id")
                            if selected_kb == "Service Requests"
                            else doc.metadata.get("doc_id")
                        )

                        if selected_kb == "Service Requests" and identifier:
                            citation_lines.append(
                                f'- **SR `{identifier}` — {title}** '
                                f'[🔗 Open SR]({source})'
                            )
                        elif identifier:
                            citation_lines.append(
                                f'- **KM Article `{identifier}` — {title}** '
                                f'[🔗 Open KB Article]({source})'
                            )
                        else:
                            citation_lines.append(
                                f'- **{title}** [🔗 Open Source]({source})'
                            )

                    citation_text = ""
                    if citation_lines:
                        citation_text = (
                            "\n\n**📚 Source References:**\n"
                            + "\n".join(citation_lines)
                        )

                    full_output = assistant_response + citation_text

            except Exception as exc:
                error_text = str(exc)
                if "503" in error_text or "UNAVAILABLE" in error_text:
                    full_output = (
                        "⚠️ The selected LLM is temporarily unavailable (HTTP 503). "
                        "Your documentation retrieval is working; please retry or select another model."
                    )
                elif "401" in error_text or "403" in error_text:
                    full_output = "⚠️ The LLM rejected the request. Please check the API key and permissions."
                elif "404" in error_text or "NOT_FOUND" in error_text:
                    full_output = "⚠️ The selected model is not available for this API key. Please select another model."
                else:
                    full_output = f"⚠️ The LLM request failed.\n\nError: `{exc}`"
                st.error(full_output)

    st.session_state.messages.append({"role": "assistant", "content": full_output})
