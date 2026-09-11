import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Optional model integrations. The app supports either Gemini or any
# OpenAI-compatible chat endpoint (OpenAI, OCI gateways, vLLM, Ollama,
# other enterprise gateways, etc.).
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Oracle Exadata RAG", layout="wide")
st.title("🤖 Oracle Exadata RAG Chatbot")

PLATFORMS = {
    "Exadata On-Prem": {
        "db_folder": "./vectordb_exadata_onprem",
        "label": "Oracle Exadata Database Machine",
        "source_urls": [
            "https://docs.oracle.com/en/engineered-systems/exadata-database-machine/dbmin/toc.htm",
            "https://docs.oracle.com/en/engineered-systems/exadata-database-machine/sagug/toc.htm",
            "https://docs.oracle.com/en/engineered-systems/exadata-database-machine/dbmso/toc.htm",
            "https://docs.oracle.com/en/engineered-systems/exadata-database-machine/dbmsq/toc.htm",
            "https://docs.oracle.com/en/engineered-systems/exadata-database-machine/dbmmn/toc.htm",
        ],
    },
    "Exascale": {
        "db_folder": "./vectordb_exascale",
        "label": "Oracle Exadata Exascale",
        "source_urls": [
            "https://docs.oracle.com/en/engineered-systems/exadata-database-machine/exscl/toc.htm",
        ],
    },
    "ExaCC": {
        "db_folder": "./vectordb_exacc",
        "label": "Oracle Exadata Database Service on Cloud@Customer",
        "source_urls": [
            "https://docs.oracle.com/en/engineered-systems/exadata-cloud-at-customer/ecccm/toc.htm",
        ],
    },
    "ExaCS": {
        "db_folder": "./vectordb_exacs",
        "label": "Oracle Exadata Database Service on Dedicated Infrastructure",
        "source_urls": [
            "https://docs.oracle.com/en/engineered-systems/exadata-cloud-service/ecscm/toc.htm",
        ],
    },
}

with st.sidebar:
    st.header("Configuration")

    selected_platform = st.selectbox(
        "Select Exadata platform",
        options=list(PLATFORMS.keys()),
        help="Answers are retrieved only from the vector index for the selected platform.",
    )

    st.divider()
    st.subheader("LLM Configuration")

    llm_provider = st.selectbox(
        "LLM provider",
        options=["OpenAI-compatible", "Google Gemini"],
        help=(
            "Use OpenAI-compatible for OpenAI or any compatible enterprise/local endpoint "
            "such as OCI gateways, vLLM, or Ollama."
        ),
    )

    if llm_provider == "OpenAI-compatible":
        llm_model = st.text_input(
            "Model name",
            value=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            help="Exact model identifier expected by your endpoint.",
        )
        llm_base_url = st.text_input(
            "Base URL (optional)",
            value=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            help=(
                "OpenAI-compatible API base URL. Examples: OpenAI, an OCI gateway, "
                "vLLM, or Ollama."
            ),
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
            value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            help="Example: gemini-2.5-flash",
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

    platform = PLATFORMS[selected_platform]
    st.divider()
    st.caption(f"Knowledge base: {platform['label']}")
    st.caption("Documentation roots used during ingestion:")
    for source_url in platform["source_urls"]:
        st.markdown(f"- [{source_url}]({source_url})")

# Clear chat when the platform changes so old context is not mixed with the new platform.
if st.session_state.get("selected_platform") != selected_platform:
    st.session_state.selected_platform = selected_platform
    st.session_state.messages = []

if not llm_model.strip():
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
db_folder = platform["db_folder"]

if not os.path.exists(db_folder):
    st.error(
        f"Vector database directory '{db_folder}' was not found. "
        "Run the ingestion notebook for this platform first."
    )
    st.stop()

@st.cache_resource
def load_vector_db(folder):
    return Chroma(persist_directory=folder, embedding_function=embedding, collection_name="exadata_docs")

db = load_vector_db(db_folder)
retriever = db.as_retriever(search_kwargs={"k": 5})

@st.cache_resource
def build_llm(provider, model, api_key, base_url, temperature):
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
    f"You are an expert system administrator assistant specializing in {platform['label']}.\n"
    "Answer using only the retrieved documentation for the currently selected platform. "
    "If the documentation does not contain the answer, say that you don't know. "
    "Do not mix information from other Exadata deployment models. Keep the answer concise and technical. "
    "When useful, mention the relevant documentation title or section.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input(f"Ask a technical question about {selected_platform}..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching {selected_platform} documentation..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                assistant_response = response["answer"]
                source_docs = response.get("context", response.get("source_documents", []))

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
                    citation_lines.append(f"- [{title}]({source})")

                citation_text = ""
                if citation_lines:
                    citation_text = "\n\n**📚 Source References:**\n" + "\n".join(citation_lines)

                full_output = assistant_response + citation_text
                st.markdown(full_output)
            except Exception as exc:
                full_output = (
                    "I could not generate the answer with the configured LLM. "
                    f"Please check the model name, API key, and endpoint.\n\n**Error:** `{exc}`"
                )
                st.error(full_output)

    st.session_state.messages.append({"role": "assistant", "content": full_output})
