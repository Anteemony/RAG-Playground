import streamlit as st

def landing_page():
    st.set_page_config("Unify Demos: RAG")

    with st.sidebar:
        unify_api_key = st.text_input("Unify API Key*", type="password", placeholder="Enter Unify API Key")
        endpoint = st.text_input("Endpoint (model@provider)*", placeholder="model@provider", value="llama-2-70b-chat@anyscale")
        pdf_docs = st.file_uploader(label="Upload PDF Document(s)*", type="pdf", accept_multiple_files=True)
        st.button("Submit Document(s)")

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
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input("ask to your document anything"):
        st.chat_message("human").write(query)
        response = "we are still working on the RAG machine... be patience :D a" 
        st.chat_message("ai").write(response)
        
def main():
    landing_page()
    chat_bot()

if __name__ == "__main__":
    main()