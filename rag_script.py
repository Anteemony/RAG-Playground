from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_unify.chat_models import ChatUnify
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
        
import random, string
from pathlib import Path
from check_session import handle_file

# create directory to store user vector files
if "USER_RANDOM_FOLDER_NAME" not in st.session_state:
    st.session_state.USER_RANDOM_FOLDER_NAME = ''.join(random.choices(string.ascii_letters + string.digits, k=40))
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', st.session_state.USER_RANDOM_FOLDER_NAME)

def field_callback(field):
    st.toast(f"{field} Updated Successfully! 🎉")

def clear_history():
    if "messages" in st.session_state:
        st.session_state.messages = []
        
def ask_unify(query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(LOCAL_VECTOR_STORE_DIR.as_posix(), embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    prompt_template = '''
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question (delimited by <qn></qn>):
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {chat_history}
    </hs>
    ------
    <qn>
    {question}
    </qn>
    Answer:
    '''
    model = ChatUnify(model=st.session_state.endpoint, unify_api_key=st.session_state.unify_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])

    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        max_tokens_limit=4000,
        rephrase_question=False,
        verbose=True
    )

    response = qa_chain({"question": query, 'chat_history': st.session_state.messages}, return_only_outputs=True)

    return response["answer"]


def process_inputs():
    with st.status("Processing Document(s)"):
        if not st.session_state.unify_api_key or not st.session_state.endpoint or not st.session_state.pdf_docs:
            st.warning("Please enter the missing fields and upload your pdf document(s)")
        else:
            # Refresh message history
            st.session_state.messages = []

            st.write("Extracting Text")
            # Extract text from PDF
            text = ""
            for pdf in st.session_state.pdf_docs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()

            # Delete PDF from Session and save space
            del st.session_state["pdf_docs"]

            st.write("Splitting Text")
            # convert to text chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            text_chunks = text_splitter.split_text(text)

            st.write("Performing Vector Storage")
            # Perform vector storage
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local(LOCAL_VECTOR_STORE_DIR.as_posix())

            # delete the files when the session ends
            handle_file(LOCAL_VECTOR_STORE_DIR.as_posix())

            st.session_state.processed_input = True
            st.success('File(s) Submitted successfuly!')


def landing_page():
    st.set_page_config("Unify Demos: RAG")
    
    with st.sidebar:

        # input for Unify API Key
        st.session_state.unify_api_key = st.text_input("Unify API Key*", type="password", placeholder="Enter Unify API Key")
        # Model and provider selection 
        model_provider = {"mixtral-8x7b-instruct-v0.1": ["together-ai", "octoai", "replicate", "mistral-ai", "perplexity-ai", "anyscale", "fireworks-ai", "lepton-ai", "deepinfra", "aws-bedrock"], "llama-2-70b-chat": ["anyscale", "perplexity-ai", "together-ai", "replicate", "octoai", "fireworks-ai", "lepton-ai", "deepinfra", "aws-bedrock"], "llama-2-13b-chat": ["anyscale", "together-ai", "replicate", "octoai", "fireworks-ai", "lepton-ai", "deepinfra", "aws-bedrock"], "mistral-7b-instruct-v0.2": ["perplexity-ai", "together-ai", "mistral-ai", "replicate", "aws-bedrock", "octoai", "fireworks-ai"], "llama-2-7b-chat": ["anyscale", "together-ai", "replicate", "fireworks-ai", "lepton-ai", "deepinfra"], "codellama-34b-instruct": ["anyscale", "perplexity-ai", "together-ai", "octoai", "fireworks-ai", "deepinfra"], "gemma-7b-it": ["anyscale", "together-ai", "fireworks-ai", "lepton-ai", "deepinfra"], "mistral-7b-instruct-v0.1": ["anyscale", "together-ai", "fireworks-ai", "deepinfra"], "mixtral-8x22b-instruct-v0.1": ["mistral-ai", "together-ai", "fireworks-ai", "deepinfra"], "codellama-13b-instruct": ["together-ai", "octoai", "fireworks-ai"], "codellama-7b-instruct": ["together-ai", "octoai"], "yi-34b-chat": ["together-ai", "deepinfra"], "llama-3-8b-chat": ["together-ai", "fireworks-ai"], "llama-3-70b-chat": ["together-ai", "fireworks-ai"], "pplx-7b-chat": ["perplexity-ai"], "mistral-medium": ["mistral-ai"], "gpt-4": ["openai"], "pplx-70b-chat": ["perplexity-ai"], "gpt-3.5-turbo": ["openai"], "deepseek-coder-33b-instruct": ["together-ai"], "gemma-2b-it": ["together-ai"], "gpt-4-turbo": ["openai"], "mistral-small": ["mistral-ai"], "mistral-large": ["mistral-ai"], "claude-3-haiku": ["anthropic"], "claude-3-opus": ["anthropic"], "claude-3-sonnet": ["anthropic"]}
        model_name = st.selectbox("Select Model",options=model_provider.keys(),index=20, placeholder= "Model", args= ("Model",))
        provider_name = st.selectbox("Select a Provider",options=model_provider[model_name], placeholder="Provider", args = ("Provider",))
        st.session_state.endpoint = f"{model_name}@{provider_name}"
        
        #Document uploader
        st.session_state.pdf_docs = st.file_uploader(label="Upload PDF Document(s)*", type="pdf", accept_multiple_files=True)
        if st.button("Submit Document(s)"):
            process_inputs()

    st.title("Unify Demos: RAG Playground")
    st.text("Chat with your PDF file using the LLM of your choice")
    st.write('''
        Usage: 
        1. Input your **Unify API Key.** If you don’t have one yet, log in to the [console](https://console.unify.ai/) to get yours.
        2. Input your Endpoint i.e. **Model and Provider ID** as model@provider. You can find both in the [benchmark interface](https://unify.ai/hub).
        3. Upload your document(s) and click the Submit button
        4. Chat Away!
        ''')


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
        response = ask_unify(query)
        st.chat_message("assistant").write(response)
        st.session_state.messages.append((query, response))

        with st.sidebar:
            st.button("Clear Chat History", type="primary", on_click=clear_history)


def main():
    landing_page()
    chat_bot()


if __name__ == "__main__":
    main()
