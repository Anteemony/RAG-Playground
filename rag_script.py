"""
Main script to run the RAG Playground application.

This script initializes session keys and launches the landing page of the application.
"""

from playground import landing_page
from playground.utils import init_keys

if __name__ == "__main__":
    # Initialize all session keys firstly
    # All new session keys should be added in the init_keys() function
    init_keys()

    landing_page()
