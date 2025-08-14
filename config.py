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

# Use absolute path for database to ensure consistency across environments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "fec_master.db")

# Use absolute paths for CSV files to ensure they work in Streamlit Cloud
CSV_FILES = {
    "bad_donors": os.path.join(BASE_DIR, "Databases", "Bad Donor Master.csv"),
    "bad_legislation": os.path.join(BASE_DIR, "Databases", "Bad Legislation Master .csv"), 
    "committees": os.path.join(BASE_DIR, "Databases", "Committee Master.csv"),
    "bad_employers": os.path.join(BASE_DIR, "Databases", "Bad Employer Master.csv"),
    "bad_groups": os.path.join(BASE_DIR, "Databases", "Bad Group Master.csv"),
    "industries": os.path.join(BASE_DIR, "Databases", "Industry Master.csv"),
    "lpac": os.path.join(BASE_DIR, "Databases", "LPAC Master.csv"),
    "rga_donors_2023": os.path.join(BASE_DIR, "Databases", "2023 RGA Donors.csv"),
    "rga_donors_2024": os.path.join(BASE_DIR, "Databases", "2024 RGA Donors.csv")
}