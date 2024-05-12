from playground import st
from pathlib import Path


def session_add(key, value, is_func=False):
    if key not in st.session_state:
        if is_func:
            st.session_state[key] = value()
        else:
            st.session_state[key] = value


def init_keys():
    # Initialize session keys
    # All new session variables should be added here

    session_add("chroma_persisted", False)
    session_add("vector_selection", "FAISS")
    session_add("embedding_model", "HuggingFaceEmbeddings")
    session_add("chunk_size", 1000)
    session_add("chunk_overlap", 100)
    session_add("messages", [])
    session_add("model_temperature", 0.3)
    session_add("store", {})
    session_add("search_type", "similarity")
    session_add("k", 4)
    session_add("fetch_k", 20)
    session_add("lambda_mult", 0.5)
    session_add("score_threshold", 0.5)
    session_add("history_unaware", False)
    session_add("search_kwargs", {})


def field_callback(field):
    st.toast(f"{field} Updated Successfully!", icon="🎉")


@st.experimental_dialog("Source Code", width="large")
def generate_src():
    st.write("Get the requirements from the requirements.txt of the repository")
    st.link_button("Go to requirements",
                   "https://github.com/Anteemony/RAG-Playground", type="primary")
    code = None
    file_path = None
    base_path = Path(__file__).parent

    if st.session_state["embedding_model"] == "HuggingFaceEmbeddings":
        if st.session_state["vector_selection"] == "FAISS":
            code_path = "../playground/data/faiss_huggingface.py"
            file_path = (base_path / code_path).resolve()

        elif st.session_state["vector_selection"] == "Pinecone":
            code_path = "../playground/data/pinecone_huggingface.py"
            file_path = (base_path / code_path).resolve()

    with (open(file_path, "r") as f):
        code = f.readlines()
        code = "".join(code).replace(
            'enter_endpoint', str(st.session_state.endpoint)
        ).replace(
            'enter_model_temperature', str(st.session_state.model_temperature)
        ).replace(
            'enter_chunk_size', str(st.session_state.chunk_size)
        ).replace(
            'enter_chunk_overlap', str(st.session_state.chunk_overlap)
        ).replace(
            'enter_search_type', str(st.session_state.search_type)
        ).replace(
            'enter_search_kwargs', str(st.session_state.search_kwargs)
        )

    st.code(code, language='python')


def clear_history():
    if "store" in st.session_state:
        st.session_state.store = {}

    if "messages" in st.session_state:
        st.session_state.messages = []

def reset_slider_value(reset_dict):
    '''
    :param reset_dict: Slider key and corresponding session_state key to reset to
    The session_state key is expected to have been created in init_keys()

    :return: None
    '''

    for key, value in reset_dict.items():
        del st.session_state[value]
        init_keys()

        st.session_state[key] = st.session_state[value]