"""
This module, `home.py`, sets up the home tab of the Streamlit application.

It includes the following main functions:
- `home_tab`: This function sets up the home tab of the Streamlit application.

The module imports necessary modules and functions from `playground`.
"""

# Import necessary modules and functions 
from playground import st  # Streamlit library for creating web apps
from playground.data.widget_data import model_provider, dynamic_provider  # Provides data for widgets
from playground.document_processing import process_inputs  # Function to process inputs
from playground.utils import field_callback, clear_history  # Callback function for fields and function to clear history

def home_tab():
    """
    This function sets up the home tab of the Streamlit application.
    It sets up the input for Unify API Key, model and provider selection, document uploader, and submit and clear chat history buttons.
    """
    # Input for Unify API Key
    st.session_state.unify_api_key = st.text_input("Unify API Key*", type="password", on_change=field_callback,
                                                   placeholder="Enter Unify API Key", args=("Unify Key ",))

    # Model and provider selection
    model_name = st.selectbox("Select Model", options=model_provider.keys(), index=20, on_change=field_callback,
                              placeholder="Model", args=("Model",))

    # Enable Dynamic Routing
    if st.toggle("Enable Dynamic Routing"):
        provider_name = st.selectbox("Select a Provider", options=dynamic_provider,
                                     on_change=field_callback,
                                     placeholder="Provider", args=("Provider",))
    else:
        provider_name = st.selectbox("Select a Provider", options=model_provider[model_name],
                                     on_change=field_callback,
                                     placeholder="Provider", args=("Provider",))

    # Set the endpoint
    st.session_state.endpoint = f"{model_name}@{provider_name}"

    # Document uploader
    st.session_state.pdf_docs = st.file_uploader(label="Upload PDF Document(s)*", type="pdf",
                                                 accept_multiple_files=True)

    # Submit Button
    if st.button("Submit Document(s)", type="primary"):
        process_inputs()

    # Clear Chat History Button
    if len(st.session_state.messages) > 0:
        st.button("🧹 Clear Chat History", key="home_clear_history", on_click=clear_history)
