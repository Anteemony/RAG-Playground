from playground import landing_page
from playground.utils import init_keys

if __name__ == "__main__":
    # Initialize all session keys firstly
    # All new session keys should be added in the init_keys() function
    init_keys()

    landing_page()
