import streamlit as st


def process_inputs():
    if not st.session_state.unify_api_key or not st.session_state.endpoint or not st.session_state.pdf_docs:
        st.warning("Please enter the missing fields and upload your pdf document(s)")


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
    st.chat_input()

def main():
    landing_page()

if __name__ == "__main__":
    main()