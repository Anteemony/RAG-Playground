"""
This module, `utils.py`, provides utility functions for the Streamlit application.

It includes the following main functions:
- `session_add`: This function adds a key-value pair to the session state.
- `init_keys`: This function initializes session keys.
- `field_callback`: This function displays a toast message when a field is updated.
- `generate_src`: This function generates a source code snippet.
- `clear_history`: This function clears the chat history.

The module imports necessary modules and functions from `playground`.
"""

# Import necessary modules and functions 
from playground import st


def session_add(key, value, is_func=False):
    """
    This function adds a key-value pair to the session state.
    If `is_func` is True, it calls the function `value` and adds the result to the session state.
    """
    # Add key-value pair to session state; if is_func is True, call value as function
    if key not in st.session_state:
        if is_func:
            st.session_state[key] = value()
        else:
            st.session_state[key] = value


def init_keys():
    """
    This function initializes session keys.
    All new session variables should be added here.
    """
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
    session_add("search_type", "similarity")
    session_add("k", 4)
    session_add("fetch_k", 20)
    session_add("lambda_mult", 0.5)
    session_add("score_threshold", 0.5)


def field_callback(field):
    """
    This function displays a toast message when a field is updated.
    """
    st.toast(f"{field} Updated Successfully!", icon="🎉")


@st.experimental_dialog("Source Code🚧", width="large")
def generate_src():
    """
    This function generates a source code snippet.
    """
    code = '''
        def coming_soon():
            print("RAG Source Code!")
        '''
    st.code(code, language='python')


def clear_history():
    """
    This function clears the history stored in the session state.
    It checks if 'store' and 'messages' keys exist in the session state and if they do, it resets them.
    """
    # Check if 'store' key exists in the session state and reset it
    if "store" in st.session_state:
        st.session_state.store = {}

    # Check if 'messages' key exists in the session state and reset it
    if "messages" in st.session_state:
        st.session_state.messages = []

def reset_slider_value(reset_dict):
    """
    This function resets the value of sliders in the session state.

    :param reset_dict: A dictionary where each key-value pair represents a slider key and corresponding session_state key to reset to.
                       The session_state key is expected to have been created in init_keys() function.
    :return: None
    """
    # Iterate over each key-value pair in the reset_dict
    for key, value in reset_dict.items():
        # Delete the session_state key
        del st.session_state[value]
        
        # Initialize the session_state key again
        init_keys()

        # Reset the slider value to the session_state value
        st.session_state[key] = st.session_state[value]