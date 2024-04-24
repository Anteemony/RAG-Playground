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
    #initialize chat history
    if "messages" not in st.session_state:
        st.session_state['messages'] = []
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("Ask your document anything..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})   
    
        response = "Coming soon... RAG machine still under construction🚧"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history    
        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    landing_page()
    chat_bot()

if __name__ == "__main__":
    main()