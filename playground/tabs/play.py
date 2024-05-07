from playground import st
from playground.document_processing import process_inputs
from playground.utils import field_callback

def playground_tab():
    st.write("🚧 => Feature Coming Soon")

    with st.container(border=True):

        with st.container(border=True):
            st.write("**Conversational Bot**")
            st.write("Disable for a simple Q&A app")
            if st.toggle("Conversational Bot", value = True):
                    st.session_state.chat_memory = True
            else:
                st.session_state.chat_memory = False

        with st.expander("Vector Storage"):
            if st.toggle("Use Online Vector Storage"):
                vector_selection = st.selectbox("Select Online Vector Storage 🚧", options=["pinecone"])
                if vector_selection == "pinecone":
                    st.text_input("Pinecone API Key", type="password", disabled=True)
                    st.text_input("Pinecone Index", disabled=True)

                    # Change it back to FAISS for now
                    vector_selection = "FAISS"
            else:
                vector_selection = st.selectbox("Select Local Vector Storage", options=["FAISS"])


        with st.expander("Embedding Model"):
            embedding_model = st.selectbox("Select Embedding Model", options=["HuggingFaceEmbeddings"])

        with st.container(border=True):
            st.write("**Adjust Parameters** ")

            with st.expander("Model"):
                model_temperature = st.slider("temperature", min_value=0.0, max_value=1.0, step=0.1, value=st.session_state.model_temperature)
                st.button("Reset", on_click=lambda: None, key="model_param_reset")

            with st.expander("Text Splitter"):

                chunk_size = st.slider("chunk_size", min_value=200, max_value=10000, step=100)
                max_overlap = min(chunk_size-99, 1000)
                chunk_overlap = st.slider("Chunk Overlap", min_value=100, max_value= max_overlap, step=100)
                
                st.button("Reset", on_click=lambda: None, key="text_splitter_param_reset")

            with st.expander("Retirever 🚧"):
                st.selectbox("Search Type", options=["similarity", "mmr", "similarity_score_threshold"])
                st.slider("k")
                st.slider("max_tokens_retrieved")
                st.slider("score_threshold")
                st.slider("fetch_k")
                st.slider("lambda_mult")
                st.text_input("filter")
                st.button("Reset", on_click=lambda: None, key="retriever_param_reset")

            st.session_state.applied_config = False

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Apply Config", on_click=field_callback, args=("Configuration",), key="apply_params_config",
                         type="primary"):
                st.session_state.embedding_model = embedding_model
                st.session_state.vector_selection = vector_selection
                st.session_state.model_temperature = model_temperature
                st.session_state.chunk_size = chunk_size
                st.session_state.chunk_overlap = chunk_overlap

                if st.session_state.chat_memory == False:
                    # Clear message display history
                    st.session_state.messages = []

                st.session_state.applied_config = True

        with col2:
            st.button("Reset all🚧", on_click=lambda: None, key="all_params_reset")

        # Process Documents outside Column
        if st.session_state.applied_config:
            process_inputs()
            st.session_state.applied_config = False