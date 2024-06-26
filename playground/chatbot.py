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


def create_conversational_rag_chain(model, retriever):
    """
    Creates a conversational RAG chain. This is a question-answering (QA) system with the ability to consider historical context.

    Parameters:
    model: The model selected by the user.
    retriever: The retriever to use for fetching relevant documents.

    Returns:
    RunnableWithMessageHistory: The conversational chain that generates the answer to the query.
    """
    
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

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """
        Retrieves the chat history for a given session.

        Parameters:
        session_id (str): The ID of the session.

        Returns:
        BaseChatMessageHistory: The chat history for the provided session ID.
        """
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain


def create_qa_chain(model, retriever):
    """
    Creates a question-answering (QA) chain for a chatbot without considering historical context.

    Parameters:
    model: The model selected by the user.
    retriever: The retriever to use for fetching relevant documents.

    Returns:
    chain: it takes a user's query as input and produces a chatbot's response as output.
    """
    
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    {context}"""

    qa_prompt_no_memory = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt_no_memory)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    return chain


def get_retriever():
    """ Creates a retriever using the vector store in the session state and the selected search parameters."""

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
    """ Depending on whether the session state is history-unaware, it returns either a conversational RAG chain or a QA chain."""
    
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
    """ Takes user queries and generates responses. It writes the user query and the response to the chat window."""
    
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
