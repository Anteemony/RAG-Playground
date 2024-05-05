from playground import st
from playground.utils import generate_src


def generate_code_tab():
    st.write("Finished adjusting the parameters to fit your use case? Get your code here.")
    st.write(" Feature coming soon.. 🚧")
    st.write("**Model**: ", st.session_state.endpoint)
    st.write("**Vectorestore**: ", "FAISS (LOCAL)")
    st.write("**Embedding Model**: ", "HuggingFaceEmbeddings")

    if st.button("Generate Source Code", type="primary"):
        generate_src()