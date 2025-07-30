#!/usr/bin/env python3
"""
Rainfall Prediction Application
Main entry point for the application
"""

import streamlit as st
from ui import main_ui
from config import STREAMLIT_CONFIG, CSS_STYLES


def main():
    """
    Main function to run the Streamlit application
    """
    # Configure Streamlit page
    st.set_page_config(
        page_title=STREAMLIT_CONFIG["page_title"],
        page_icon=STREAMLIT_CONFIG["page_icon"],
        layout=STREAMLIT_CONFIG["layout"],
        initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"],
    )

    # Add custom CSS for better styling
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

    # Run the main UI
    main_ui()


if __name__ == "__main__":
    main()
