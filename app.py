"""Compatibility entry point for Streamlit Cloud.

The actual dashboard implementation lives in streamlit_app.py. Keeping this
small wrapper lets an existing Streamlit Cloud app continue working even if its
configured main file is app.py.
"""

import streamlit_app  # noqa: F401
