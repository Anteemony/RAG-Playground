"""
This module, `chatbot`, provides functionalities to create a conversational RAG chain.

It includes the following main functions:
- `create_conversational_rag_chain`: This function creates a conversational RAG chain using a given model and a history-aware retriever.

The module imports necessary modules and functions from `langchain`, `langchain_core`, `langchain_unify`, `playground`, and `langchain_community`.

This module is part of the larger project that aims to build a chatbot using RAG (Retrieval-Augmented Generation) model.
"""

# Import necessary modules and functions for the chatbot module
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_unify.chat_models import ChatUnify
from playground import st
from playground.document_processing import process_inputs, format_docs, output_chunks
from langchain_community.chat_message_histories import ChatMessageHistory

def create_conversational_rag_chain(model, history_aware_retriever):
    """
    Creates a conversational RAG chain.

    Args:
        model: The model to be used for the RAG chain.
        history_aware_retriever: The retriever that is aware of the chat history.

    Returns:
        A RunnableWithMessageHistory object that represents the conversational RAG chain.
    """
    # Define the system prompt for the contextualized question
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    

    # Create a ChatPromptTemplate object for the contextualized question
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),  # System prompt
            MessagesPlaceholder("chat_history"),    # Placeholder for chat history
            ("human", "{input}"),   # User input
        ]
    )

    # Create a chain of documents for the history-aware retriever# Create a history-aware retriever with the model, formatted documents, and the contextualized question prompt
    history_aware_retriever = create_history_aware_retriever(
        model,  # The model to be used 
        retriever | format_docs,     # The retriever and the formatted documents 
        contextualize_q_prompt    # The contextualized question prompt
    )

    # Define the prompt for the QA system
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    {context}"""

    # Create a ChatPromptTemplate object for the QA system
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a chain of documents for the QA system  
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    
    # Create a retrieval chain with the history-aware retriever and the QA chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """Gets the chat history for a given session.

        Args:
            session_id: The ID of the session.

        Returns:
            The chat history for the session.
        """
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    # Create a RunnableWithMessageHistory object for the RAG chain
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

def create_qa_chain(model, retriever):
    """Creates a QA chain.

    Args:
        model: The model to be used for the QA chain.
        retriever: The retriever to be used for the QA chain.

    Returns:
        A chain that represents the QA system.
    """
    # Define the prompt for the QA system
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    {context}"""

    # Create a ChatPromptTemplate object for the QA system
    qa_prompt_no_memory = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )
    # Create a chain of documents for the QA system
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt_no_memory)
    
    # Create a retrieval chain with the retriever and the QA chain
    chain = create_retrieval_chain(retriever, question_answer_chain)

    return chain


def get_retriever():
    '''
    Set the corresponding search keyword arguments
    based on the search type
    :return: VectorStoreRetriever
    '''

    if st.session_state.search_type == "similarity":
        st.session_state.search_kwargs = {"k": st.session_state.k}

    elif st.session_state.search_type == "similarity_score_threshold":
        st.session_state.search_kwargs = {
            "k": st.session_state.k,
            "score_threshold": st.session_state.score_threshold
        }

    elif st.session_state.search_type == "mmr":
        st.session_state.search_kwargs = {
            "k": st.session_state.k,
            "fetch_k": st.session_state.fetch_k,
            "lambda_mult": st.session_state.lambda_mult
        }

    retriever = st.session_state.vector_store.as_retriever(
        search_type=st.session_state.search_type,
        search_kwargs=st.session_state.search_kwargs
    )

    return retriever


def ask_unify():
    if "vector_store" not in st.session_state:
        process_inputs()

    retriever = get_retriever()

    model = ChatUnify(
        model=st.session_state.endpoint,
        unify_api_key=st.session_state.unify_api_key,
        temperature=st.session_state.model_temperature
    )

    # Return the appropriate chain
    if not st.session_state.history_unaware:
        return create_conversational_rag_chain(model, retriever)
    else:
        return create_qa_chain(model, retriever)


def chat_bot():
    if query := st.chat_input("Ask your document anything...", key="query"):

        if "processed_input" not in st.session_state:
            st.warning("Please input your details in the sidebar first")
            return

        st.chat_message("human").write(query)

        rag_engine = ask_unify()

        response = st.chat_message("assistant").write_stream(
            output_chunks(rag_engine, query)
        )

        if not st.session_state.history_unaware:
            st.session_state.messages.append((query, response))
