from PyPDF2 import PdfReader
from playground import st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


model_provider = {
                "mixtral-8x7b-instruct-v0.1": ["together-ai", "octoai", "replicate", "mistral-ai", "perplexity-ai",
                                               "anyscale", "fireworks-ai", "lepton-ai", "deepinfra", "aws-bedrock"],
                "llama-2-70b-chat": ["anyscale", "perplexity-ai", "together-ai", "replicate", "octoai", "fireworks-ai",
                                     "lepton-ai", "deepinfra", "aws-bedrock"],
                "llama-2-13b-chat": ["anyscale", "together-ai", "replicate", "octoai", "fireworks-ai", "lepton-ai",
                                     "deepinfra", "aws-bedrock"],
                "mistral-7b-instruct-v0.2": ["perplexity-ai", "together-ai", "mistral-ai", "replicate", "aws-bedrock",
                                             "octoai", "fireworks-ai"],
                "llama-2-7b-chat": ["anyscale", "together-ai", "replicate", "fireworks-ai", "lepton-ai", "deepinfra"],
                "codellama-34b-instruct": ["anyscale", "perplexity-ai", "together-ai", "octoai", "fireworks-ai",
                                           "deepinfra"],
                "gemma-7b-it": ["anyscale", "together-ai", "fireworks-ai", "lepton-ai", "deepinfra"],
                "mistral-7b-instruct-v0.1": ["anyscale", "together-ai", "fireworks-ai", "deepinfra"],
                "mixtral-8x22b-instruct-v0.1": ["mistral-ai", "together-ai", "fireworks-ai", "deepinfra"],
                "codellama-13b-instruct": ["together-ai", "octoai", "fireworks-ai"],
                "codellama-7b-instruct": ["together-ai", "octoai"], "yi-34b-chat": ["together-ai", "deepinfra"],
                "llama-3-8b-chat": ["together-ai", "fireworks-ai"], "llama-3-70b-chat": ["together-ai", "fireworks-ai"],
                "pplx-7b-chat": ["perplexity-ai"], "mistral-medium": ["mistral-ai"], "gpt-4": ["openai"],
                "pplx-70b-chat": ["perplexity-ai"], "gpt-3.5-turbo": ["openai"],
                "deepseek-coder-33b-instruct": ["together-ai"], "gemma-2b-it": ["together-ai"], "gpt-4-turbo": ["openai"],
                "mistral-small": ["mistral-ai"], "mistral-large": ["mistral-ai"], "claude-3-haiku": ["anthropic"],
                "claude-3-opus": ["anthropic"], "claude-3-sonnet": ["anthropic"]
}

dynamic_provider = ["lowest-input-cost", "lowest-output-cost", "lowest-itl", "lowest-ttft", "highest-tks-per-sec"]

@st.cache_data
def extract_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


@st.cache_data
def faiss_vector_storage(text_chunks):
    vector_store = None

    if st.session_state.embedding_model == "HuggingFaceEmbeddings":
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    return vector_store

def process_inputs():

    if not st.session_state.unify_api_key or not st.session_state.endpoint or not st.session_state.pdf_docs:
        st.warning("Please enter the missing fields and upload your pdf document(s)")
    else:
        with st.status("Processing Document(s)"):

            st.write("Extracting Text")
            # Extract text from PDF
            text = extract_pdf(st.session_state.pdf_docs)

            st.write("Splitting Text")
            # convert to text chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )
            text_chunks = text_splitter.split_text(text)

            st.write("Performing Vector Storage")

            # Perform vector storage
            if st.session_state.vector_selection == "FAISS":
                st.session_state.vector_store = faiss_vector_storage(text_chunks)

            st.session_state.processed_input = True
            st.success('File(s) Submitted successfully!')

def format_docs(docs):
    return [doc for doc in docs]


def output_chunks(chain, query):
    for chunk in chain.stream(
            {"input": query},
            config={"configurable": {"session_id": "abc123"}}
    ):
        if "answer" in chunk.keys():
            yield chunk["answer"]
