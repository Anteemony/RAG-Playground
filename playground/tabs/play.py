from playground import st
from playground.document_processing import process_inputs
from playground.utils import field_callback, reset_slider_value, clear_history
from playground.data.widget_data import model_reset_dict, splitter_reset_dict, retriever_reset_dict, model_max_context_limit


def playground_tab():
    st.write("**Welcome to Playground!**.🛝")
    st.write("Adjust the application settings and parameters to suite your use case.",
             "Don't forget to click the **Apply Configuration** button at the bottom after editing")
    st.write("⚠️: If you end up getting errors, readjust the parameters or click the Reset Buttons!")

    with st.container(border=True):

        with st.expander("Vector Storage"):
            if st.toggle("Use Online Vector Storage"):
                vector_selection = st.selectbox("Select Online Vector Storage", options=["Pinecone"])
                if vector_selection == "Pinecone":
                    st.session_state.pinecone_api_key = st.text_input("Pinecone API Key", type="password")
                    st.session_state.pinecone_index = st.text_input("Pinecone Index")
                    st.write("⚠️: The index records will be cleared and started afresh")
            else:
                vector_selection = st.selectbox("Select Local Vector Storage", options=["FAISS"])

        with st.expander("Embedding Model"):
            embedding_model = st.selectbox("Select Embedding Model", options=["HuggingFaceEmbeddings"])

        with st.container(border=True):
            st.write("**Adjust Parameters** ")

            with st.expander("Model"):
                model_temperature = st.slider("temperature", key="slider_model_temperature", min_value=0.0,
                                              max_value=1.0, step=0.1, value=st.session_state.model_temperature)

                st.button("Reset", on_click=reset_slider_value, args=(model_reset_dict,), key="model_param_reset")

            with st.expander("Text Splitter"):
                # set ma chunk_size token to model context limit
                max_token = model_max_context_limit[st.session_state.get("endpoint").split("@")[0]]
                chunk_size = st.slider("chunk_size", key="slider_chunk_size", min_value=200, max_value=max_token, step=100,
                                       value=st.session_state.chunk_size)
                max_overlap = min(chunk_size - 99, 1000)
                chunk_overlap = st.slider("Chunk Overlap", key="slider_chunk_overlap", min_value=100,
                                          max_value=max_overlap, step=100, value=st.session_state.chunk_overlap)

                st.button("Reset", on_click=reset_slider_value, args=(splitter_reset_dict,),
                          key="text_splitter_param_reset")

            with st.expander("Retirever"):
                search_type = st.selectbox("Search Type", options=["similarity", "mmr", "similarity_score_threshold"],
                                           help="Defines the type of search that the Retriever should perform.")
                k = st.slider(
                    "k",
                    key="slider_k",
                    help="Amount of documents to return (Default: 4)",
                    value=st.session_state.k
                )

                if search_type == "similarity_score_threshold":
                    score_threshold = st.slider(
                        "score_threshold",
                        key="slider_score_threshold",
                        help="Minimum relevance threshold for similarity_score_threshold",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.1,
                        value=st.session_state.score_threshold,
                    )

                if search_type == "mmr":
                    fetch_k = st.slider(
                        "fetch_k",
                        key="slider_fetch_k",
                        help="Amount of documents to pass to MMR algorithm (Default: 20)",
                        value=st.session_state.fetch_k
                    )
                    lambda_mult = st.slider(
                        "lambda_mult",
                        key="slider_lambda_mult",
                        help="Diversity of results returned by MMR; 1 for minimum diversity and 0 for maximum. (Default: 0.5)",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.1,
                        value=st.session_state.lambda_mult
                    )

                # TODO
                # with st.container(border=True):
                #     st.markdown("filter", help="Filter by document metadata")
                #
                #     col1, col2 = st.columns([0.5, 1])
                #     with col1:
                #         st.text_input(label="key")
                #
                #     with col2:
                #         st.text_input(label="value")
                #
                # with st.container(border=True):
                #     st.markdown("Set Max Tokens", help="The retrieved document tokens will be checked and reduced below this limit.")
                #     st.slider("max_tokens_retrieved")

                st.button("Reset", on_click=reset_slider_value, args=(retriever_reset_dict,),
                          key="retriever_param_reset")

            st.session_state.applied_config = False

        with st.expander("Extras"):
            with st.container(border=True):
                st.write("**Conversational Bot**")
                st.write("Enable for a history aware chatbot")
                st.write("Disable for a simple Q&A app with no history attached.")

                if st.toggle("Conversational Bot", value=st.session_state.chat_memory):
                    chat_memory = True
                else:
                    chat_memory = False

        if st.button("Apply Configuration", on_click=field_callback, args=("Configuration",), key="apply_params_config",
                     type="primary"):
            st.session_state.embedding_model = embedding_model
            st.session_state.vector_selection = vector_selection
            st.session_state.model_temperature = model_temperature
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            st.session_state.chat_memory = chat_memory

            if not st.session_state.chat_memory:
                # Clear message display history
                st.session_state.messages = []

            # Retriever
            st.session_state.search_type = search_type
            st.session_state.k = k

            if search_type == "similarity_score_threshold":
                st.session_state.score_threshold = score_threshold

            if search_type == "mmr":
                st.session_state.fetch_k = fetch_k
                st.session_state.lambda_mult = lambda_mult

            st.session_state.applied_config = True

        # Process Documents outside Column
        if st.session_state.applied_config:
            process_inputs()
            st.session_state.applied_config = False

    # Clear Chat History Button
    if len(st.session_state.messages) > 0:
        st.button("🧹 Clear Chat History", key="play_clear_history", on_click=clear_history)
