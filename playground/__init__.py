"""
This module, `__init__`, sets up the landing page of the Streamlit application.

It includes the following main functions:
- `landing_page`: This function sets up the landing page of the Streamlit application.

The module imports necessary modules and functions from `streamlit` and `playground`.
"""

# Import necessary modules and functions
import streamlit as st
from playground.chatbot import chat_bot
from playground.tabs.home import home_tab
from playground.tabs.generate import display_parameters
from playground.tabs.play import playground_tab

def landing_page():
    """
    This function sets up the landing page of the Streamlit application.
    It sets the page configuration, displays the title and instructions,
    shows the chat messages, and sets up the sidebar with tabs for different sections of the application.
    """

    # Set the page configuration
    st.set_page_config("RAG Playground", page_icon="🎉")

    # Display the title and instructions
    st.title("Langchain RAG Playground 🛝")
    st.text("Chat with your PDF file using the LLM of your choice")
    st.write('''
            Usage: 
            1. Input your **Unify API Key.** If you don’t have one yet, log in to the [console](https://console.unify.ai/) to get yours.
            2. Select the **Model** and endpoint provider of your choice from the drop down. You can find both model and provider information in the [benchmark interface](https://unify.ai/hub).
            3. Upload your document(s) and click the Submit button
            4. Chat Away!
            ''')

    # Show the chat messages
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('assistant').write(message[1])

    # Set up the sidebar with tabs for different sections of the application
    with st.sidebar:
        tab1, tab2, tab3 = st.tabs(["🏠Home", "🛝Playground", "🎉Generate Code"])

        # Display the home tab
        with tab1:
            home_tab()

        # Display the playground tab
        with tab2:
            playground_tab()

        # Display the generate code tab
        with tab3:
            display_parameters()

    # Call the chat_bot function
    chat_bot()
