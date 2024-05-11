from playground import st
from playground.utils import generate_src


def generate_code_tab():
    st.write("Finished adjusting the parameters to fit your use case? Get your code here.")
    st.write(" Feature coming soon.. 🚧")

    st.write("**Parameters**")
    with st.container(border=True):
        st.write("**Vector store**: ", st.session_state.vector_selection)
        st.write("**Embedding Model**: ", st.session_state.embedding_model)

        with st.container(border=True):
            st.write("**Endpoint**: ")
            st.text("model: " + str(st.session_state.endpoint.split("@")[0]))
            st.text("provider: " + str(st.session_state.endpoint.split("@")[1]))
            st.text("temperature: " + str(st.session_state.model_temperature))

        with st.container(border=True):
            st.write("**Text Splitter**")
            st.text("chunk_size: " + str(st.session_state.chunk_size))
            st.text("chunk_overlap: " + str(st.session_state.chunk_overlap))

        with st.container(border=True):
            st.write("**Retriever**")
            st.text("search_type: " + str(st.session_state.search_type))
            st.text("k: " + str(st.session_state.k))

            if st.session_state.search_type == "similarity_score_threshold":
                st.text("similarity_score_threshold: " + str(st.session_state.score_threshold))

            if st.session_state.search_type == "mmr":
                st.text("fetch_k: " + str(st.session_state.fetch_k))
                st.text("lambda_mult: " + str(st.session_state.lambda_mult))






    if st.button("Generate Source Code", type="primary"):
        generate_src()