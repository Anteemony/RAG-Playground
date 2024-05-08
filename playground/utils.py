from playground import st


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
    session_add("chat_memory", True)


def field_callback(field):
    st.toast(f"{field} Updated Successfully!", icon="🎉")


@st.experimental_dialog("Source Code🚧", width="large")
def generate_src():
    code = '''
        def coming_soon():
            print("RAG Source Code!")
        '''
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