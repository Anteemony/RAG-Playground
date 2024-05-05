from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_unify.chat_models import ChatUnify
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st


def field_callback(field):
    st.toast(f"{field} Updated Successfully!", icon="🎉")

@st.experimental_dialog("Source Code", width="large")
def generate_src():
    code = '''
        def hello():
            print("RAG Source Code!")
        '''
    st.code(code, language='python')

def clear_history():
    if "store" in st.session_state:
        st.session_state.store = {}

    if "messages" in st.session_state:
        st.session_state.messages = []


@st.cache_data
def extract_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


@st.cache_data
def perform_vector_storage(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def format_docs(docs):
    return [doc for doc in docs]


def output_chunks(chain, query):
    for chunk in chain.stream(
            {"input": query},
            config={"configurable": {"session_id": "abc123"}}
    ):
        if "answer" in chunk.keys():
            yield chunk["answer"]


def ask_unify():
    if "vector_store" not in st.session_state:
        process_inputs()

    retriever = st.session_state.vector_store.as_retriever()

    if "model_temperature" not in st.session_state:
        st.session_state.model_temperature = 0.3

    model = ChatUnify(
        model=st.session_state.endpoint,
        unify_api_key=st.session_state.unify_api_key,
        temperature=st.session_state.model_temperature
    )

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        model, retriever | format_docs, contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    if "store" not in st.session_state:
        st.session_state.store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain


def process_inputs():
    if not st.session_state.unify_api_key or not st.session_state.endpoint or not st.session_state.pdf_docs:
        st.warning("Please enter the missing fields and upload your pdf document(s)")
    else:
        with st.status("Processing Document(s)"):
            # Refresh message history
            st.session_state.messages = []

            st.write("Extracting Text")
            # Extract text from PDF
            text = extract_pdf(st.session_state.pdf_docs)

            if "chunk_size" not in st.session_state:
                st.session_state.chunk_size = 1000
            if "chunk_overlap" not in st.session_state:
                st.session_state.chunk_overlap = 100

            st.write("Splitting Text")
            # convert to text chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )
            text_chunks = text_splitter.split_text(text)

            st.write("Performing Vector Storage")
            # Perform vector storage
            st.session_state.vector_store = perform_vector_storage(text_chunks)

            st.session_state.processed_input = True
            st.success('File(s) Submitted successfully!')


def landing_page():
    st.set_page_config("Unify Demos: RAG")

    st.title("Langchain RAG Playground 🛝")
    st.text("Chat with your PDF file using the LLM of your choice")
    st.write('''
                    Usage: 
                    1. Input your **Unify API Key.** If you don’t have one yet, log in to the [console](https://console.unify.ai/) to get yours.
                    2. Select the **Model** and endpoint provider of your choice from the drop down. You can find both model and provider information in the [benchmark interface](https://unify.ai/hub).
                    3. Upload your document(s) and click the Submit button
                    4. Chat Away!
                    ''')

    with st.sidebar:
        tab1, tab2, tab3 = st.tabs(["🏠Home", "🛝Playground", "🎉Generate Code"])

        with tab1:
            # input for Unify API Key
            st.session_state.unify_api_key = st.text_input("Unify API Key*", type="password", on_change=field_callback,
                                                           placeholder="Enter Unify API Key", args=("Unify Key ",))
            # Model and provider selection
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
                "claude-3-opus": ["anthropic"], "claude-3-sonnet": ["anthropic"]}
            dynamic_provider = ["lowest-input-cost", "lowest-output-cost", "lowest-itl", "lowest-ttft", "highest-tks-per-sec"]
            model_name = st.selectbox("Select Model", options=model_provider.keys(), index=20, on_change=field_callback,
                                      placeholder="Model", args=("Model",))
            if st.toggle("Enable Dynamic Routing"):
                provider_name = st.selectbox("Select a Provider", options=dynamic_provider,
                                             on_change=field_callback,
                                             placeholder="Provider", args=("Provider",))
            else:
                provider_name = st.selectbox("Select a Provider", options=model_provider[model_name],
                                             on_change=field_callback,
                                             placeholder="Provider", args=("Provider",))
            st.session_state.endpoint = f"{model_name}@{provider_name}"

            # Document uploader
            st.session_state.pdf_docs = st.file_uploader(label="Upload PDF Document(s)*", type="pdf",
                                                         accept_multiple_files=True)
            if st.button("Submit Document(s)"):
                process_inputs()

            # Clear Chat History Button
            if "messages" in st.session_state:
                if len(st.session_state.messages) > 0:
                    st.button("Clear Chat History", type="primary", on_click=clear_history)

        with tab2:
            st.write("Feature Coming Soon🚧")

            with st.expander("Vector Storage"):
                if st.toggle("Use Local Vector Storage"):
                    st.selectbox("Select Local Vector Storage", options=["FAISS", "Chroma"])
                else:
                    online_vectore_storage = st.selectbox("Select Online Vector Storage", options=["pinecone", "pinecone_similar"])
                    if online_vectore_storage == "pinecone":
                        st.text_input("Pincecone API Key", type="password")

            with st.expander("Emebedding Model"):
                st.selectbox("Select Embedding Model", options=["HuggingFaceEmbeddings", "ChatOpenAIEmbeddings", "GPT4AllEmbeddings"])

            with st.container(border=True):

                st.write("**Adjust Parameters** ")

                with st.expander("Prompt Template"):
                    st.text_input("System Prompt")
                    st.text_input("Hub link")
                    st.button("Reset", on_click=lambda: None, key="prompt_template_reset")

                with st.expander("Model"):
                    model_temperature = st.slider("temperature", min_value=0.0, max_value=1.0, step=0.1)
                    st.button("Reset", on_click=lambda: None, key="model_param_reset")

                with st.expander("Text Splitter"):
                    chunk_size = st.slider("chunk_size", min_value=100, max_value=10000, step=100)
                    chunk_overlap = st.slider("chunk_overlap", min_value=100, max_value=1000, step=100)
                    st.button("Reset", on_click=lambda: None, key="text_splitter_param_reset")

                with st.expander("Retirever"):
                    st.selectbox("Search Type", options=["similarity", "mmr", "similarity_score_threshold"])
                    st.slider("k")
                    st.slider("max_tokens_retrieved")
                    st.slider("score_threshold")
                    st.slider("fetch_k")
                    st.slider("lambda_mult")
                    st.text_input("filter")
                    st.button("Reset", on_click=lambda: None, key="retriever_param_reset")

                st.session_state.applied_config=False

                col1, col2 = st.columns([1, 1])
        
                with col1:
                    if st.button("Apply Config", on_click=field_callback, args=("Parameters", ), key="apply_params_config", type="primary"):
                        st.session_state.model_temperature = model_temperature
                        st.session_state.chunk_size = chunk_size
                        st.session_state.chunk_overlap = chunk_overlap
                        st.session_state.applied_config = True

                with col2:
                    st.button("Reset all", on_click=lambda: None, key="all_params_reset")

                # Process Documents outside Column
                if st.session_state.applied_config:
                    process_inputs()
                    st.session_state.applied_config = False


        with tab3:
            st.write("Finished adjusting the parameters to fit your use case? Get your code here.")
            st.write(" Feature coming soon.. 🚧")
            st.write("**Model**: ", st.session_state.endpoint)
            st.write("**Vectorestore**: ", "FAISS (LOCAL)")
            st.write("**Embedding Model**: ", "HuggingFaceEmbeddings")

            if st.button("Generate Source Code", type="primary"):
                generate_src()


def chat_bot():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('assistant').write(message[1])
    #
    if query := st.chat_input("Ask your document anything...", key="query"):

        if "processed_input" not in st.session_state:
            st.warning("Please input your details in the sidebar first")
            return

        st.chat_message("human").write(query)

        conversational_rag_chain = ask_unify()

        response = st.chat_message("assistant").write_stream(
            output_chunks(conversational_rag_chain, query)
        )

        st.session_state.messages.append((query, response))


def main():
    landing_page()
    chat_bot()


if __name__ == "__main__":
    main()
