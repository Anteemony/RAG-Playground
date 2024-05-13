"""
This module, `generate.py`, sets up the generate code tab of the Streamlit application.

It includes the following main functions:
- `generate_code_tab`: This function sets up the generate code tab of the Streamlit application.

The module imports necessary modules and functions from `playground`.
"""

# Import necessary modules and functions 
from playground import st  # Streamlit library for creating web apps
from playground.utils import generate_src  # Function to generate source code

def generate_code_tab():
    """
    This function sets up the generate code tab of the Streamlit application.
    It displays the parameters and a button to generate the source code.
    """
    # Display the parameters
    st.write("Finished adjusting the parameters to fit your use case? Get your code here.")
    st.write(" Feature coming soon.. 🚧")
    st.write("**Model**: ", st.session_state.endpoint)
    st.write("**Vectorestore**: ", "FAISS (LOCAL)")
    st.write("**Embedding Model**: ", "HuggingFaceEmbeddings")

    # Button to generate the source code
    if st.button("Generate Source Code", type="primary"):
        generate_src()