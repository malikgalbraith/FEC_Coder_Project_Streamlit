import sqlite3
import pandas as pd
import os
from config import DATABASE_PATH, CSV_FILES

def create_database():
    """Create SQLite database and import all CSV files"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create tables for each database
    create_tables(cursor)
    
    # Import CSV data
    import_csv_data(conn)
    
    # Create indexes for fast lookups
    create_indexes(cursor)
    
    conn.commit()
    conn.close()
    print(f"Database created successfully at {DATABASE_PATH}")

def create_tables(cursor):
    """Create all necessary tables"""
    
    # Bad Donors table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bad_donors (
            id INTEGER PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            state TEXT,
            full_key TEXT,
            name_key TEXT,
            laststate_key TEXT,
            affiliation TEXT
        )
    ''')
    
    # Bad Legislation table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bad_legislation (
            id INTEGER PRIMARY KEY,
            committee_id TEXT,
            committee_fec_name TEXT,
            associated_candidate TEXT,
            life_conception_cosponsorships INTEGER,
            life_conception_118th INTEGER,
            life_conception_117th INTEGER,
            life_conception_116th INTEGER,
            life_conception_115th INTEGER,
            life_conception_114th INTEGER,
            life_conception_113th INTEGER,
            life_conception_112th INTEGER,
            voted_against_certifying INTEGER,
            rsc_member INTEGER,
            voted_repeal_aca INTEGER,
            pact_act INTEGER,
            mifepristone_brief INTEGER,
            election_overturn_brief INTEGER,
            voted_against_infrastructure INTEGER,
            voted_against_pro_act INTEGER,
            voted_tax_cuts INTEGER,
            voted_against_inflation_reduction INTEGER
        )
    ''')
    
    # Committee Master table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS committees (
            id INTEGER PRIMARY KEY,
            cmte_id TEXT UNIQUE,
            cmte_nm TEXT,
            tres_nm TEXT,
            cmte_st1 TEXT,
            cmte_st2 TEXT,
            cmte_city TEXT,
            cmte_st TEXT,
            cmte_zip TEXT,
            cmte_dsgn TEXT,
            cmte_tp TEXT,
            cmte_pty_affiliation TEXT,
            cmte_filing_freq TEXT,
            org_tp TEXT,
            connected_org_nm TEXT,
            cand_id TEXT
        )
    ''')
    
    # Bad Employers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bad_employers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            flag TEXT
        )
    ''')
    
    # Bad Groups table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bad_groups (
            id INTEGER PRIMARY KEY,
            committee_id TEXT,
            committee_name TEXT,
            flag TEXT
        )
    ''')
    
    # Industry Master table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS industries (
            id INTEGER PRIMARY KEY,
            committee_id TEXT,
            committee_name TEXT,
            org_type TEXT,
            smaller_categories TEXT,
            larger_categories TEXT,
            source TEXT,
            connected_organization TEXT,
            country_of_origin TEXT
        )
    ''')
    
    # LPAC Master table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lpac (
            id INTEGER PRIMARY KEY,
            committee_id TEXT,
            short_name TEXT,
            official_lpac_name TEXT,
            pac_sponsor_district TEXT,
            updated_date TEXT
        )
    ''')

def import_csv_data(conn):
    """Import data from CSV files into database tables"""
    
    # Bad Donors
    if os.path.exists(CSV_FILES["bad_donors"]):
        df = pd.read_csv(CSV_FILES["bad_donors"])
        # Handle variable number of columns, drop empty ones
        df = df.dropna(axis=1, how='all')  # Drop completely empty columns
        expected_cols = ['first_name', 'last_name', 'state', 'full_key', 'name_key', 'laststate_key', 'affiliation']
        df.columns = expected_cols[:len(df.columns)]  # Assign columns up to available count
        # Add missing columns if needed
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ''
        df = df[expected_cols]  # Reorder to expected structure
        df.to_sql('bad_donors', conn, if_exists='replace', index=False)
        print(f"Imported {len(df)} bad donors")
    
    # Bad Legislation
    if os.path.exists(CSV_FILES["bad_legislation"]):
        df = pd.read_csv(CSV_FILES["bad_legislation"])
        # Clean column names - handle variable number of columns
        expected_cols = [
            'committee_id', 'committee_fec_name', 'associated_candidate', 
            'life_conception_cosponsorships', 'life_conception_118th', 'life_conception_117th',
            'life_conception_116th', 'life_conception_115th', 'life_conception_114th',
            'life_conception_113th', 'life_conception_112th', 'voted_against_certifying',
            'rsc_member', 'voted_repeal_aca', 'pact_act', 'mifepristone_brief',
            'election_overturn_brief', 'voted_against_infrastructure', 'voted_against_pro_act',
            'voted_tax_cuts', 'voted_against_inflation_reduction'
        ]
        df.columns = expected_cols[:len(df.columns)]
        # Add missing columns with default values
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0 if 'life_conception' in col or 'voted' in col or col in ['rsc_member', 'pact_act'] else ''
        df = df[expected_cols]
        df.to_sql('bad_legislation', conn, if_exists='replace', index=False)
        print(f"Imported {len(df)} bad legislation records")
    
    # Committee Master
    if os.path.exists(CSV_FILES["committees"]):
        df = pd.read_csv(CSV_FILES["committees"])
        expected_cols = [
            'cmte_id', 'cmte_nm', 'tres_nm', 'cmte_st1', 'cmte_st2', 'cmte_city',
            'cmte_st', 'cmte_zip', 'cmte_dsgn', 'cmte_tp', 'cmte_pty_affiliation',
            'cmte_filing_freq', 'org_tp', 'connected_org_nm', 'cand_id'
        ]
        df.columns = expected_cols[:len(df.columns)]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ''
        df = df[expected_cols]
        df.to_sql('committees', conn, if_exists='replace', index=False)
        print(f"Imported {len(df)} committees")
    
    # Bad Employers
    if os.path.exists(CSV_FILES["bad_employers"]):
        df = pd.read_csv(CSV_FILES["bad_employers"])
        expected_cols = ['name', 'flag']
        df.columns = expected_cols[:len(df.columns)]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ''
        df = df[expected_cols]
        df.to_sql('bad_employers', conn, if_exists='replace', index=False)
        print(f"Imported {len(df)} bad employers")
    
    # Bad Groups
    if os.path.exists(CSV_FILES["bad_groups"]):
        df = pd.read_csv(CSV_FILES["bad_groups"])
        expected_cols = ['committee_id', 'committee_name', 'flag']
        df.columns = expected_cols[:len(df.columns)]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ''
        df = df[expected_cols]
        df.to_sql('bad_groups', conn, if_exists='replace', index=False)
        print(f"Imported {len(df)} bad groups")
    
    # Industry Master
    if os.path.exists(CSV_FILES["industries"]):
        df = pd.read_csv(CSV_FILES["industries"])
        expected_cols = [
            'committee_id', 'committee_name', 'org_type', 'smaller_categories',
            'larger_categories', 'source', 'connected_organization', 'country_of_origin'
        ]
        df.columns = expected_cols[:len(df.columns)]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ''
        df = df[expected_cols]
        df.to_sql('industries', conn, if_exists='replace', index=False)
        print(f"Imported {len(df)} industry records")
    
    # LPAC Master
    if os.path.exists(CSV_FILES["lpac"]):
        df = pd.read_csv(CSV_FILES["lpac"])
        expected_cols = ['committee_id', 'short_name', 'official_lpac_name', 'pac_sponsor_district', 'updated_date']
        df.columns = expected_cols[:len(df.columns)]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ''
        df = df[expected_cols]
        df.to_sql('lpac', conn, if_exists='replace', index=False)
        print(f"Imported {len(df)} LPAC records")

def create_indexes(cursor):
    """Create indexes for fast lookups"""
    
    # Bad Donors indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bad_donors_full_key ON bad_donors(full_key)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bad_donors_name_key ON bad_donors(name_key)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bad_donors_laststate_key ON bad_donors(laststate_key)')
    
    # Committee indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_committees_id ON committees(cmte_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_committees_name ON committees(cmte_nm)')
    
    # Bad Legislation indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bad_legislation_id ON bad_legislation(committee_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bad_legislation_name ON bad_legislation(committee_fec_name)')
    
    # Bad Groups indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bad_groups_id ON bad_groups(committee_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bad_groups_name ON bad_groups(committee_name)')
    
    # Industry indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_industries_id ON industries(committee_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_industries_name ON industries(committee_name)')
    
    # LPAC indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_lpac_id ON lpac(committee_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_lpac_name ON lpac(official_lpac_name)')
    
    # Bad Employers indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bad_employers_name ON bad_employers(name)')
    
    print("Created all database indexes")

def get_connection():
    """Get database connection"""
    return sqlite3.connect(DATABASE_PATH)

if __name__ == "__main__":
    create_database()