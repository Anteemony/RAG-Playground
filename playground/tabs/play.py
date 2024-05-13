# Import necessary modules and functions
from playground import st
from playground.document_processing import process_inputs
from playground.utils import field_callback, reset_slider_value, clear_history
from playground.data.widget_data import model_reset_dict, splitter_reset_dict, retriever_reset_dict, model_max_context_limit

def playground_tab():
    """
    Main function for the Playground tab in the user interface.

    This function provides a user interface for adjusting application settings and parameters,
    and for selecting and configuring various components of the system.

    The user can select the vector storage (either online or local), the embedding model,
    and adjust various parameters for the model.

    The function uses the Streamlit library for the user interface.
    """
    # Welcome message
    st.write("**Welcome to Playground!**.🛝")
    # Instructions for the user
    st.write("Adjust the application settings and parameters to suite your use case.",
             "Don't forget to click the **Apply Configuration** button at the bottom after editing")
    # Error handling message
    st.write("⚠️: If you end up getting errors, readjust the parameters or click the Reset Buttons!")

    # Container for vector storage selection
    with st.container(border=True):
        # Selection list for Vector Storage and API input if required
        with st.expander("Vector Storage"):
            # Toggle for online vector storage
            if st.toggle("Use Online Vector Storage"):
                # Dropdown for online vector storage selection
                vector_selection = st.selectbox("Select Online Vector Storage", options=["Pinecone"])
                # If Pinecone is selected, ask for API key and index
                if vector_selection == "Pinecone":
                    st.session_state.pinecone_api_key = st.text_input("Pinecone API Key", type="password")  # Input for Pinecone API key
                    st.session_state.pinecone_index = st.text_input("Pinecone Index")  # Input for Pinecone index
                    st.write("⚠️: The index records will be cleared and started afresh")
            else:
                # Dropdown for local vector storage selection
                vector_selection = st.selectbox("Select Local Vector Storage", options=["FAISS"])
                
        # Selection list for Embedding Model 
        with st.expander("Embedding Model"):
            # Dropdown for embedding model selection
            embedding_model = st.selectbox("Select Embedding Model", options=["HuggingFaceEmbeddings"],
                                           help="Select the embedding model to use for the application. (Default: HuggingFaceEmbeddings)")

        # RAG Parameters container
        with st.container(border=True):
            st.write("**Adjust Parameters** ")

            # Slider for model Temperature
            with st.expander("Model"):
                # Slider for adjusting model temperature
                model_temperature = st.slider("temperature", key="slider_model_temperature", min_value=0.0,
                                              max_value=1.0, step=0.1, value=st.session_state.model_temperature,
                                              help="The temperature parameter for the model. (Default: 0.3)")

                # Reset button for model parameters
                st.button("Reset", on_click=reset_slider_value, args=(model_reset_dict,), key="model_param_reset")

            # Text splitting parameters
            with st.expander("Text Splitter"):
                # Set maximum token limit to model context
                max_token = model_max_context_limit[st.session_state.get("endpoint").split("@")[0]]  # get max token limit of selected LLM model

                # Set chunk maximum size to split the document (in Tokens)
                chunk_size = st.slider("chunk_size", key="slider_chunk_size", min_value=200, max_value=max_token, step=100,
                                       value=st.session_state.chunk_size,
                                       help="The maximum size of each chunk in tokens. (Default: 1000)")
                max_overlap = min(chunk_size - 99, 1000)  # max_overlap < chunk_size to avoid infinite loops while processing documents
                chunk_overlap = st.slider("Chunk Overlap", key="slider_chunk_overlap", min_value=0,
                                          max_value=max_overlap, step=100, 
                                          value=st.session_state.chunk_overlap,
                                          help="The number of tokens to overlap between chunks. (Default: 100)")

                # Reset button for text splitter parameters
                st.button("Reset", on_click=reset_slider_value, args=(splitter_reset_dict,),
                          key="text_splitter_param_reset")

            # Select the retriever search type between:
            # Similarity, MMR or Similarirty Score Treshold
            with st.expander("Retirever"):
                search_type = st.selectbox("Search Type", options=["similarity", "mmr", "similarity_score_threshold"],
                                           help="Defines the type of search that the Retriever should perform.")

                # Define K as the number of documents the retriever will return
                k = st.slider(
                    "k",
                    key="slider_k",
                    help="Amount of documents to return (Default: 4)",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.k
                )

                # Parameter tuning for Similarity_score_threshold algorithm
                if search_type == "similarity_score_threshold":
                    score_threshold = st.slider(
                        "score_threshold",
                        key="slider_score_threshold",
                        help="Minimum relevance threshold for a document to be returned.",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.1,
                        value=st.session_state.score_threshold,
                    )

                # Parameter tuning for MMR algorithm
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

                # Reset button for retriever parameters
                st.button("Reset", on_click=reset_slider_value, args=(retriever_reset_dict,),
                          key="retriever_param_reset")

            # Set the applied_config state to False
            st.session_state.applied_config = False

        
            # Check if 'chat_memory' is in the session state, if not initialize it
            if "chat_memory" not in st.session_state:
                st.session_state.chat_memory = []
                
            # Toggle to enable/disable chat history memory and RAG conversational feature    
            with st.expander("Chat history"):
                # Toggle for enabling/disabling conversational bot
                if st.toggle("Conversational Bot", value=st.session_state.chat_memory,
                             help="Enable for a history aware chatbot. Disable for a simple Q&A app with no history attached."):
                    chat_memory = True
                else:
                    history_unaware = False

            # Apply configuration button
            if st.button("Apply Configuration", on_click=field_callback, args=("Configuration",), key="apply_params_config",
                         type="primary"):
                # Set session state variables
                st.session_state.embedding_model = embedding_model
                st.session_state.vector_selection = vector_selection
                st.session_state.model_temperature = model_temperature
                st.session_state.chunk_size = chunk_size
                st.session_state.chunk_overlap = chunk_overlap
                st.session_state.history_unaware = history_unaware

                # Clear message display history if chat memory is not enabled
                if st.session_state.history_unaware:
                    st.session_state.messages = []

                # Set retriever session state variables
                st.session_state.search_type = search_type
                st.session_state.k = k

                # Set specific retriever session state variables based on search type
                if search_type == "similarity_score_threshold":
                    st.session_state.score_threshold = score_threshold

                if search_type == "mmr":
                    st.session_state.fetch_k = fetch_k
                    st.session_state.lambda_mult = lambda_mult

                # Set applied_config state to True
                st.session_state.applied_config = True

            # Process Documents outside Column
            if st.session_state.applied_config:
                process_inputs()
                st.session_state.applied_config = False

            # Clear Chat History Button
            if len(st.session_state.messages) > 0:
                st.button("🧹 Clear Chat History", key="play_clear_history", on_click=clear_history)
                
