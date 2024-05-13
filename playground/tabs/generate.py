"""
This module, `generate.py`, is responsible for displaying the parameters and a button to generate the source code.

It includes the following main sections:
- Display of parameters
- Containers for Endpoint, Text Splitter, and Retriever parameters

The module uses the Streamlit library for creating the web interface and session state for storing parameters.
"""

# Import necessary modules and functions
import streamlit as st


def display_parameters():
    """
    This function displays the parameters and a button to generate the source code.
    """
    # Display the parameters
    st.write("Finished adjusting the parameters to fit your use case? Get your code here.")

    # Container for parameters
    with st.container(border=True):
        st.write("**Parameters**")
        
        # Container for Endpoint parameters
        with st.container(border=True):
            st.write("**Endpoint**: ")
            st.text("model: " + str(st.session_state.endpoint.split("@")[0]))
            st.text("provider: " + str(st.session_state.endpoint.split("@")[1]))
            st.text("temperature: " + str(st.session_state.model_temperature))

        # Container for Text Splitter parameters
        with st.container(border=True):
            st.write("**Text Splitter**")
            st.text("chunk_size: " + str(st.session_state.chunk_size))
            st.text("chunk_overlap: " + str(st.session_state.chunk_overlap))

        # Container for Retriever parameters
        with st.container(border=True):
            st.write("**Retriever**")
            st.write("*Vector store*: ", st.session_state.vector_selection)
            st.write("*Embedding Model*: ", st.session_state.embedding_model)
            st.write("*Retriever Keywords*: {")
            st.text("search_type: " + str(st.session_state.search_type))
            st.text("k: " + str(st.session_state.k))

            # Display additional parameters based on the search type
            if st.session_state.search_type == "similarity_score_threshold":
                st.text("similarity_score_threshold: " + str(st.session_state.score_threshold))

            if st.session_state.search_type == "mmr":
                st.text("fetch_k: " + str(st.session_state.fetch_k))
                st.text("lambda_mult: " + str(st.session_state.lambda_mult))

            st.write("}")

    # Button to generate the source code
    if st.button("Generate Source Code", type="primary"):
        generate_src()