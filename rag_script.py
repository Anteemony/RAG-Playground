from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_unify.chat_models import ChatUnify
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from streamlit.runtime.state import session_state
import streamlit as st

def ask_unify(query):

		embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
		vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
		retriever = vectorstore.as_retriever()

		prompt_template = '''Use the provided context to answer the question \nContext: {context} \nQuestion: {question} \n\n Answer'''
		model = ChatUnify(model=st.session_state.endpoint, unify_api_key=st.session_state.unify_api_key)
		prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
		qa_chain = ConversationalRetrievalChain.from_llm(
			llm=model,
			retriever=retriever,
			return_source_documents=True
		)

		response = qa_chain({"question": query, 'chat_history': st.session_state.messages}, return_only_outputs=True)

		return response["answer"]

def process_inputs():
		if not st.session_state.unify_api_key or not st.session_state.endpoint or not st.session_state.pdf_docs:
	@@ -40,7 +46,7 @@ def process_inputs():
			text_chunks = text_splitter.split_text(text)

			# Perform vector storage
			embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
			vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
			vector_store.save_local("faiss_index")

	@@ -84,7 +90,7 @@ def chat_bot():
			st.chat_message("human").write(query)
			response = ask_unify(query) 
			st.chat_message("assistant").write(response)
			st.session_state.messages.append((query, response))

def main():
    landing_page()
