from PyPDF2 import PdfReader
from playground import st
from playground.data.widget_data import model_provider, dynamic_provider
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import os


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

def pinecone_vector_storage(text_chunks):
    vector_store = None

    os.environ['PINECONE_API_KEY'] = st.session_state.pinecone_api_key

    if st.session_state.embedding_model == "HuggingFaceEmbeddings":
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = PineconeVectorStore.from_texts(
            text_chunks, 
            embedding=embeddings, 
            index_name=st.session_state.pinecone_index
        )
    
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
            
            elif st.session_state.vector_selection == "Pinecone":
                st.session_state.vector_store = pinecone_vector_storage(text_chunks)

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
