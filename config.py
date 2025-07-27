import os
from dotenv import load_dotenv

load_dotenv()

# Try to get API key from Streamlit secrets first, then environment variables
try:
    import streamlit as st
    FEC_API_KEY = st.secrets.get("FEC_API_KEY", os.getenv("FEC_API_KEY"))
except (ImportError, FileNotFoundError, KeyError):
    FEC_API_KEY = os.getenv("FEC_API_KEY")
FEC_BASE_URL = "https://api.open.fec.gov/v1"
DATABASE_PATH = "fec_master.db"

CSV_FILES = {
    "bad_donors": "Databases/Bad Donor Master.csv",
    "bad_legislation": "Databases/Bad Legislation Master .csv", 
    "committees": "Databases/Committee Master.csv",
    "bad_employers": "Databases/Bad Employer Master.csv",
    "bad_groups": "Databases/Bad Group Master.csv",
    "industries": "Databases/Industry Master.csv",
    "lpac": "Databases/LPAC Master.csv"
}