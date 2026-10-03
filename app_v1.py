import os
from pathlib import Path
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Optional model integrations. The app supports either Gemini or any
# OpenAI-compatible chat endpoint (OpenAI, OCI gateways, vLLM, Ollama,
# other enterprise gateways, etc.).
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

st.set_page_config(
    page_title="Exadata Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Exadata Chatbot")

APP_DIR = Path(__file__).resolve().parent

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

if "selected_kb" not in st.session_state:
    st.session_state.selected_kb = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configuration")

    selected_kb = st.selectbox(
        "Select knowledge source",
        options=list(KNOWLEDGE_BASES.keys()),
        help="Search only the vector index for the selected knowledge source.",
    )

    st.divider()
    st.subheader("LLM Configuration")

    llm_provider = st.selectbox(
        "LLM provider",
        options=[
            "No LLM - Documentation Search",
            "OpenAI-compatible",
            "Google Gemini",
        ],
        help=(
            "Select No LLM when you do not have an API key. "
            "OpenAI-compatible supports OpenAI and other compatible endpoints."
        ),
    )

    if llm_provider == "No LLM - Documentation Search":
        llm_model = ""
        llm_base_url = ""
        llm_api_key = ""
        st.info(
            "🔎 No LLM mode: retrieve relevant content from the selected source "
            "and show its source links without generating an answer."
        )

    elif llm_provider == "OpenAI-compatible":
        llm_model = st.text_input(
            "Model name",
            value=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            help="Exact model identifier expected by your endpoint.",
        )
        llm_base_url = st.text_input(
            "Base URL (optional)",
            value=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            help="OpenAI-compatible API base URL.",
        )
        llm_api_key = st.text_input(
            "API key",
            value=os.getenv("LLM_API_KEY", ""),
            type="password",
            help="Leave blank only when your endpoint does not require authentication.",
        )

    else:
        llm_model = st.text_input(
            "Gemini model",
            value=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            help="Example: gemini-3.5-flash",
        )
        llm_base_url = ""
        llm_api_key = st.text_input(
            "Gemini API key",
            value=os.getenv("GOOGLE_API_KEY", ""),
            type="password",
        )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Lower values are better for precise documentation-based answers.",
    )

    kb = KNOWLEDGE_BASES[selected_kb]
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

# Clear chat when the knowledge source changes so its history stays source-specific.
if st.session_state.selected_kb != selected_kb:
    st.session_state.selected_kb = selected_kb
    st.session_state.messages = []

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
        This source contains Oracle internal Knowledge/KM or Service Request content.
        Use it only within the authorized internal team and do not share responses
        outside approved users.
        </div>
        """,
        unsafe_allow_html=True,
    )

if llm_provider not in ("No LLM - Documentation Search",) and not llm_model.strip():
    st.info("Please enter an LLM model name in the sidebar to begin.")
    st.stop()

if llm_provider == "Google Gemini" and not llm_api_key.strip():
    st.info("Please enter your Gemini API key in the sidebar to begin.")
    st.stop()

@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

embedding = load_embedding()
db_folder = kb["db_folder"]

if not os.path.exists(db_folder):
    st.error(
        f"Vector database directory '{db_folder}' was not found. "
        "Run the ingestion notebook or script for this knowledge source first."
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
    db = load_vector_db(db_folder, kb["collection_name"])
except Exception as exc:
    st.error(f"Unable to load Chroma database `{db_folder}`. Error: {exc}")
    st.stop()

# Fetch enough chunks to rank and group internal articles/SRs before display.
retriever = db.as_retriever(search_kwargs={"k": 60})


def is_missing_collection_error(exc):
    message = str(exc).lower()
    return "collection" in message and "does not exist" in message


def refresh_vector_db():
    """Discard stale Streamlit and Chroma handles and reconnect to the index."""
    global db, retriever

    load_vector_db.clear()
    # Chroma also caches its system by persist_directory. Clearing only the
    # Streamlit resource leaves that system pointing at a replaced collection.
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except ImportError:
        from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()
    db = load_vector_db(db_folder, kb["collection_name"])
    retriever = db.as_retriever(search_kwargs={"k": 60})


def _query_terms(query):
    return {
        term.lower().strip(".,:;!?()[]{}\"'")
        for term in query.split()
        if len(term.strip(".,:;!?()[]{}\"'")) >= 3
    }


def _exact_search_terms(query):
    """Find hostnames, SR numbers, and Oracle error/bug IDs for exact lookup."""
    import re

    pattern = (
        r"\b(?:4-\d{10}|(?:ORA|BUG)-?\s*\d+|"
        r"[A-Z][A-Z0-9_-]*\d[A-Z0-9_-]*)\b"
    )
    terms = []
    for value in re.findall(pattern, query, flags=re.IGNORECASE):
        value = value.strip(".,:;!?()[]{}\"'")
        if value and value.lower() not in {term.lower() for term in terms}:
            terms.append(value)
    return terms


def retrieve_sr_documents(query):
    """Combine semantic results with exact text matches for IDs and hostnames."""
    docs = retriever.invoke(query)
    exact_terms = _exact_search_terms(query)
    exact_search_failed = False

    for term in exact_terms:
        # Case variants help with hostnames copied from logs or entered by hand.
        for variant in dict.fromkeys((term, term.lower(), term.upper())):
            try:
                docs.extend(
                    db.similarity_search(
                        query,
                        k=60,
                        where_document={"$contains": variant},
                    )
                )
            except Exception as exc:
                if is_missing_collection_error(exc):
                    raise
                exact_search_failed = True

    unique_docs = []
    seen = set()
    for doc in docs:
        key = (
            doc.metadata.get("sr_id"),
            doc.metadata.get("source"),
            doc.page_content,
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    if exact_search_failed:
        st.caption("Exact host/SR lookup was unavailable; showing semantic search results.")
    return unique_docs


def retrieve_source_documents(query):
    """Retry once with a fresh Chroma handle if the cached collection was replaced."""
    for attempt in range(2):
        try:
            if selected_kb == "Service Requests":
                return retrieve_sr_documents(query)
            return retriever.invoke(query)
        except Exception as exc:
            if attempt or not is_missing_collection_error(exc):
                raise
            refresh_vector_db()
            st.info("The vector index changed while the app was running; reloaded it and retried.")


def rank_km_documents(query, docs, max_articles=5):
    """Rank Oracle Knowledge articles so one article's chunks do not dominate."""
    query_terms = _query_terms(query)
    intent_terms = {
        "minimum", "supported", "support", "versions", "version", "database",
        "db", "grid", "upgrade", "patch", "release", "compatibility",
        "requirement", "requirements",
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
        score = (
            5.0 * sum(1 for term in query_terms if term in title)
            + 3.0 * sum(1 for term in intent_terms if term in title)
            + sum(1 for term in query_terms if term in content)
            + min(sum(1 for term in intent_terms if term in content), 8) * 0.25
        )

        if doc_id not in grouped:
            grouped[doc_id] = {"score": score, "docs": [doc]}
        else:
            grouped[doc_id]["docs"].append(doc)
            grouped[doc_id]["score"] = max(grouped[doc_id]["score"], score)

    ranked = sorted(grouped.items(), key=lambda item: item[1]["score"], reverse=True)
    return ranked[:max_articles]


def build_km_context(ranked_articles, max_chunks_per_article=4):
    """Choose the longest useful chunks from each selected KM article."""
    context_docs = []
    for _, article in ranked_articles:
        article_docs = sorted(
            article["docs"],
            key=lambda doc: len(doc.page_content or ""),
            reverse=True,
        )
        context_docs.extend(article_docs[:max_chunks_per_article])
    return context_docs


def rank_sr_documents(query, docs, max_articles=5):
    """Rank SR chunks by title/content and group the results by SR number."""
    import re

    query_terms = _query_terms(query)
    intent_terms = {
        "issue", "problem", "error", "failure", "patch", "upgrade", "bug",
        "ora", "rac", "asm", "acfs", "dataguard", "backup", "performance",
        "vulnerability", "swap", "network", "dns",
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
        score = (
            7.0 * sum(1 for term in query_terms if term in title)
            + 1.5 * sum(1 for term in query_terms if term in content)
            + 1.5 * sum(1 for term in intent_terms if term in content)
        )

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
    """Choose query-relevant chunks from each selected service request."""
    query_terms = _query_terms(query)
    context_docs = []

    for _, article in ranked_articles:
        scored_docs = []
        for doc in article["docs"]:
            title = (doc.metadata.get("title") or "").lower()
            content = (doc.page_content or "").lower()
            score = (
                5.0 * sum(1 for term in query_terms if term in title)
                + 2.0 * sum(1 for term in query_terms if term in content)
            )
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        context_docs.extend(
            doc for _, doc in scored_docs[:max_chunks_per_article]
        )

    return context_docs


def build_sr_search_fallback(query, docs, max_records=3, max_chars_per_record=900):
    """Show retrieved SR excerpts when the LLM cannot confirm a direct answer."""
    query_lower = query.lower()
    is_staffing_query = (
        any(word in query_lower for word in ("support", "available", "standby", "engineer"))
        and any(word in query_lower for word in ("change", "maintenance", "patch", "upgrade"))
    )
    grouped = {}
    for doc in docs:
        sr_id = (
            doc.metadata.get("sr_id")
            or doc.metadata.get("source")
            or doc.metadata.get("title")
            or str(id(doc))
        )
        grouped.setdefault(sr_id, []).append(doc)

    if not grouped:
        if is_staffing_query:
            return (
                "\n\nNo matching SR passages were retrieved. Historical SRs also cannot confirm "
                "current staffing. For a live support check, include the planned date, time and "
                "timezone, plus the SR or change request ID."
            )
        return (
            "\n\nNo matching SR passages were retrieved. Add the exact error text, "
            "an ORA or BUG number, the operation that failed, and the Exadata/GI version."
        )

    lines = [
        "\n\n**Closest matching Service Requests**",
        "These are historical examples; they may describe different symptoms or environments.",
    ]
    for sr_id, record_docs in list(grouped.items())[:max_records]:
        doc = record_docs[0]
        title = doc.metadata.get("title") or "Untitled Service Request"
        source = doc.metadata.get("source")
        excerpts = [
            item.page_content.strip()
            for item in record_docs[:2]
            if item.page_content and item.page_content.strip()
        ]
        excerpt = "\n\n".join(excerpts)
        if len(excerpt) > max_chars_per_record:
            excerpt = excerpt[:max_chars_per_record].rstrip() + "..."

        reference = f"SR `{sr_id}`"
        if source:
            reference = f"[{reference} — Open SR]({source})"
        lines.append(f"\n- **{reference}: {title}**\n  {excerpt}")

    if is_staffing_query:
        lines.append(
            "\n\nHistorical SRs cannot confirm current staffing. For a live support check, "
            "include the planned date, time and timezone, plus the SR or change request ID."
        )
    else:
        lines.append(
            "\n\nFor a more precise match, include the exact error message, an ORA/BUG number, "
            "the operation that failed, and the relevant host or software version."
        )
    return "\n".join(lines)

@st.cache_resource
def build_llm(provider, model, api_key, base_url, temperature):
    if provider == "No LLM - Documentation Search":
        return None

    if provider == "Google Gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
        )

    kwargs = {
        "model": model,
        "temperature": temperature,
    }
    if api_key.strip():
        kwargs["api_key"] = api_key
    if base_url.strip():
        kwargs["base_url"] = base_url.rstrip("/")

    return ChatOpenAI(**kwargs)

try:
    llm = build_llm(
        llm_provider,
        llm_model.strip(),
        llm_api_key.strip(),
        llm_base_url.strip(),
        temperature,
    )
except Exception as exc:
    st.error(f"Unable to initialize the selected LLM: {exc}")
    st.stop()

system_prompt = (
    f"You are an expert technical assistant specializing in {kb['label']}. "
    f"The selected knowledge source is {selected_kb}.\n\n"
    "Answer using only retrieved content from the selected knowledge source. "
    "Do not answer only 'I don't know.' If the retrieved content does not establish a direct answer, "
    "say what is missing and ask one focused follow-up question. For broad Service Request queries, "
    "summarize the closest retrieved SRs as historical examples even when none confirms a fix. "
    "Do not introduce unsupported information or mix knowledge sources. "
    "Keep the answer concise and technical. When useful, mention the document title or identifier. "
    "For Service Requests, distinguish customer statements, Oracle support actions, findings, "
    "workarounds, and resolution status. Do not turn one historical SR statement into a general "
    "Oracle recommendation. Preserve important dates and qualifiers. Historical SRs do not confirm "
    "live support staffing or availability. For a planned change, only say support is available if "
    "retrieved content explicitly confirms the assignment, date, and time. Otherwise say the retrieved "
    "records do not confirm staffing; do not claim the host or its records are absent from the full "
    "index. Ask for the planned date/time and SR number when needed.\n\nContext:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input(f"Ask a technical question about {selected_kb}..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching {selected_kb}..."):
            try:
                retrieved_docs = retrieve_source_documents(user_query)

                if selected_kb == "Oracle Knowledge":
                    ranked_articles = rank_km_documents(
                        user_query, retrieved_docs, max_articles=3
                    )
                    source_docs = build_km_context(
                        ranked_articles,
                        max_chunks_per_article=(2 if llm is None else 4),
                    )
                elif selected_kb == "Service Requests":
                    ranked_articles = rank_sr_documents(
                        user_query, retrieved_docs, max_articles=5
                    )
                    source_docs = build_sr_context(
                        user_query,
                        ranked_articles,
                        max_chunks_per_article=(2 if llm is None else 4),
                    )
                else:
                    source_docs = retrieved_docs[:8]

                if llm_provider == "No LLM - Documentation Search":
                    if not source_docs:
                        full_output = "No relevant documentation was found."
                    else:
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
                            source = doc.metadata.get("source")
                            title = (
                                doc.metadata.get("title")
                                or doc.metadata.get("document_title")
                                or (source.rsplit("/", 1)[-1] if source else "Unknown source")
                            )
                            identifier = (
                                doc.metadata.get("sr_id")
                                if selected_kb == "Service Requests"
                                else doc.metadata.get("doc_id")
                            )
                            content = "\n\n".join(
                                item.page_content.strip()
                                for item in docs_for_source[:2]
                                if item.page_content and item.page_content.strip()
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

                            if source:
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
                                    parts.append(f"**Source:** [🔗 Open Source]({source})\n\n---")
                        full_output = "\n".join(parts)
                else:
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    assistant_response = question_answer_chain.invoke(
                        {"input": user_query, "context": source_docs}
                    )

                    normalized_response = assistant_response.strip().lower()
                    inconclusive_markers = (
                        "i don't know",
                        "i do not know",
                        "does not contain any information",
                        "no information regarding",
                        "couldn't find",
                        "could not find",
                        "not present in retrieved",
                        "not found in retrieved",
                    )
                    if (
                        selected_kb == "Service Requests"
                        and any(marker in normalized_response for marker in inconclusive_markers)
                    ):
                        assistant_response += build_sr_search_fallback(user_query, source_docs)

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
                                f'- **SR `{identifier}` — {title}** [🔗 Open SR]({source})'
                            )
                        elif identifier:
                            citation_lines.append(
                                f'- **KM Article `{identifier}` — {title}** '
                                f'[🔗 Open KB Article]({source})'
                            )
                        else:
                            citation_lines.append(f"- [{title}]({source})")

                    citation_text = ""
                    if citation_lines:
                        citation_text = "\n\n**📚 Source References:**\n" + "\n".join(citation_lines)

                    full_output = assistant_response + citation_text

                st.markdown(full_output)
            except Exception as exc:
                full_output = (
                    "I could not search this knowledge source or generate the answer. "
                    "Please check that its vector database is available and that the "
                    f"selected LLM is configured correctly.\n\n**Error:** `{exc}`"
                )
                st.error(full_output)

    st.session_state.messages.append({"role": "assistant", "content": full_output})
