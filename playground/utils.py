from playground import st


def field_callback(field):
    st.toast(f"{field} Updated Successfully!", icon="🎉")


@st.experimental_dialog("Source Code", width="large")
def generate_src():
    code = '''
        def hello():
            print("RAG Source Code!")
        '''
    st.code(code, language='python')


def clear_history():
    if "store" in st.session_state:
        st.session_state.store = {}

    if "messages" in st.session_state:
        st.session_state.messages = []
