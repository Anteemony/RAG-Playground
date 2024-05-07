import streamlit as st
from playground.chatbot import chat_bot
from playground.tabs.home import home_tab
from playground.tabs.play import playground_tab
from playground.tabs.generate import generate_code_tab


def landing_page():
    st.set_page_config("RAG Playground", page_icon="🎉")

    st.title("Langchain RAG Playground 🛝")
    st.text("Chat with your PDF file using the LLM of your choice")
    st.write('''
            Usage: 
            1. Input your **Unify API Key.** If you don’t have one yet, log in to the [console](https://console.unify.ai/) to get yours.
            2. Select the **Model** and endpoint provider of your choice from the drop down. You can find both model and provider information in the [benchmark interface](https://unify.ai/hub).
            3. Upload your document(s) and click the Submit button
            4. Chat Away!
            ''')

    # show messages
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('assistant').write(message[1])

    with st.sidebar:
        tab1, tab2, tab3 = st.tabs(["🏠Home", "🛝Playground", "🎉Generate Code"])

        with tab1:
            home_tab()

        with tab2:
            playground_tab()

        with tab3:
            generate_code_tab()

    chat_bot()
