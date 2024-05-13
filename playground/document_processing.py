"""
This module, `document_processing`, provides functionalities to process documents for the chatbot.

It includes the following main functions:
- `extract_pdf`: This function extracts text from PDF documents.
- `faiss_vector_storage`: This function creates a FAISS vector store from the given text chunks.
- `pinecone_vector_storage`: This function creates a Pinecone vector store from the given text chunks.

The module imports necessary modules and functions from `PyPDF2`, `playground`, `langchain`, `langchain_community`, `langchain_pinecone`, and `os`.
"""

# Import necessary modules and functions 
from PyPDF2 import PdfReader
from playground import st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import os


@st.cache_data      # Decorator to cache the function results
def extract_pdf(pdf_docs):    
    """Extracts text from PDF documents.

    Args:
        pdf_docs (list): A list of PDF documents.

    Returns:
        str: The extracted text from the PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text   # Returns the extracted text from the PDF documents   


@st.cache_data    # Decorator to cache the function results 
def faiss_vector_storage(text_chunks):
    """Creates a FAISS vector store from the given text chunks.

    Args:
        text_chunks (list): A list of text chunks to be vectorized.

    Returns:
        FAISS: A FAISS vector store.
    """
    vector_store = None  # Initialize the vector store as None

    # Check if the selected embedding model is HuggingFaceEmbeddings
    if st.session_state.embedding_model == "HuggingFaceEmbeddings":
        # Create HuggingFaceEmbeddings with the specified model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Create a FAISS vector store from the text chunks using the embeddings
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    return vector_store # Returns the FAISS vector store    


def pinecone_vector_storage(text_chunks):
    """Creates a Pinecone vector store from the given text chunks.

    Args:
        text_chunks (list): A list of text chunks to be vectorized.

    Returns:
        PineconeVectorStore: A Pinecone vector store.
    """
    vector_store = None # Initialize the vector store as None

    # Set the Pinecone API key from the session state
    os.environ['PINECONE_API_KEY'] = st.session_state.pinecone_api_key

    # Check if the selected embedding model is HuggingFaceEmbeddings
    if st.session_state.embedding_model == "HuggingFaceEmbeddings":
        # Create HuggingFaceEmbeddings with the specified model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        try:
            # Clear existing index data if there's any
            PineconeVectorStore.from_existing_index(
                index_name=st.session_state.pinecone_index,
                embedding=embeddings
            ).delete(delete_all=True)
        except Exception as e:
            print("The index is empty")
        finally:
            # Create a Pinecone vector store from the text chunks using the embeddings
            vector_store = PineconeVectorStore.from_texts(
                text_chunks,
                embedding=embeddings,
                index_name=st.session_state.pinecone_index
            )
    
    return vector_store # Returns the Pinecone vector store 
    

def process_inputs():
    """Processes the user inputs and performs vector storage.

    This function checks if all necessary fields are filled and PDF documents are uploaded.
    If yes, it extracts text from the PDFs, splits the text into chunks, and performs vector storage using either FAISS or Pinecone.

    Returns:
        None
    """
    # Check if all necessary fields are filled and PDF documents are uploaded
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

def format_docs(docs):      #Formats the given documents into a list
    return [doc for doc in docs]


def output_chunks(chain, query):
    """Generates answers for the given query from the chain.

    Args:
        chain (Chain): The chain to generate answers from.
        query (str): The query to generate answers for.

    Yields:
        str: The generated answer.
    """
    # Iterate over the chunks generated by the chain for the given query
    for chunk in chain.stream(
            {"input": query},
            config={"configurable": {"session_id": "abc123"}}
    ):
        # If the chunk contains an answer, yield it
        if "answer" in chunk.keys():
            yield chunk["answer"]
