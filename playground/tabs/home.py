from playground import st
from playground.data.widget_data import model_provider, dynamic_provider
from playground.document_processing import process_inputs
from playground.utils import field_callback, clear_history


def home_tab():
    # input for Unify API Key
    st.session_state.unify_api_key = st.text_input("Unify API Key*", type="password", on_change=field_callback,
                                                   placeholder="Enter Unify API Key", args=("Unify Key ",))
    # Model and provider selection
    model_name = st.selectbox("Select Model", options=model_provider.keys(), index=20, on_change=field_callback,
                              placeholder="Model", args=("Model",))

    if st.toggle("Enable Dynamic Routing"):
        provider_name = st.selectbox("Select a Provider", options=dynamic_provider,
                                     on_change=field_callback,
                                     placeholder="Provider", args=("Provider",))
    else:
        provider_name = st.selectbox("Select a Provider", options=model_provider[model_name],
                                     on_change=field_callback,
                                     placeholder="Provider", args=("Provider",))

    st.session_state.endpoint = f"{model_name}@{provider_name}"

    # Document uploader
    st.session_state.pdf_docs = st.file_uploader(label="Upload PDF Document(s)*", type="pdf",
                                                 accept_multiple_files=True)

    # Submit Button
    if st.button("Submit Document(s)"):
        process_inputs()

    # Clear Chat History Button
    if "messages" in st.session_state:
        if len(st.session_state.messages) > 0:
            st.button("Clear Chat History", type="primary", on_click=clear_history)
