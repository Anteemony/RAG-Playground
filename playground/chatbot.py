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

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
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
    # Simplified RAG system without using chat history
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\
    {context}"""
        
    qa_prompt_no_memory = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )
    # Simple retriever setup, similar but without using history
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt_no_memory)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return chain


def ask_unify():
    if "vector_store" not in st.session_state:
        process_inputs()
        
    retriever = st.session_state.vector_store.as_retriever()

    model = ChatUnify(
        model=st.session_state.endpoint,
        unify_api_key=st.session_state.unify_api_key,
        temperature=st.session_state.model_temperature
    )

    # Return the approriate chain        
    if st.session_state.chat_memory == True:
        return create_conversational_rag_chain(model, retriever)
    else:
        return create_qa_chain (model, retriever)


def chat_bot():
    if query := st.chat_input("Ask your document anything...", key="query"):

        if "processed_input" not in st.session_state:
            st.warning("Please input your details in the sidebar first")
            return

        st.chat_message("human").write(query)

        Rag_engine = ask_unify()

        response = st.chat_message("assistant").write_stream(
            output_chunks(Rag_engine, query)
        )

        if st.session_state.chat_memory:
            st.session_state.messages.append((query, response))