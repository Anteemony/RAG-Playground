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
		st.warning("Please enter the missing fields and upload your pdf document(s)")
	else:
		# Refresh message history
		st.session_state.messages = []
		
		# Extract text from PDF
		text = ""
		for pdf in st.session_state.pdf_docs:
			pdf_reader = PdfReader(pdf)
			for page in pdf_reader.pages:
				text += page.extract_text()

		# convert to text chunks
		text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=200)
		text_chunks = text_splitter.split_text(text)

		# Perform vector storage
		embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
		vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
		vector_store.save_local("faiss_index")

		st.session_state.processed_input = True


def landing_page():
	st.set_page_config("Unify Demos: RAG")

	with st.sidebar:
		unify_api_key = st.text_input("Unify API Key*", type="password", placeholder="Enter Unify API Key", key="unify_api_key")
		endpoint = st.text_input("Endpoint (model@provider)*", placeholder="model@provider", value="llama-2-70b-chat@anyscale", key="endpoint")
		pdf_docs = st.file_uploader(label="Upload PDF Document(s)*", type="pdf", accept_multiple_files=True, key="pdf_docs")
		st.button("Submit Document(s)", on_click=process_inputs)
		
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
        
def main():
    landing_page()
    chat_bot()

if __name__ == "__main__":
    main()
