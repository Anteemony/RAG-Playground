#https://discuss.streamlit.io/t/detecting-user-exit-browser-tab-closed-session-end/62066

import threading, shutil

from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime import get_instance

def handle_file(USER_FOLDER_STRING):
    thread = threading.Timer(interval=30, function=handle_file, args=(USER_FOLDER_STRING,))

    # insert context to the current thread, needed for
    # getting session specific attributes like st.session_state

    add_script_run_ctx(thread)

    # context is required to get session_id of the calling
    # thread (which would be the script thread)
    ctx = get_script_run_ctx()

    runtime = get_instance()  # this is the main runtime, contains all the sessions

    if runtime.is_active_session(session_id=ctx.session_id):
        # Session is running
        thread.start()
    else:
        # Session is not running, Do what you want to do on user exit here
        shutil.rmtree(USER_FOLDER_STRING)
        return
