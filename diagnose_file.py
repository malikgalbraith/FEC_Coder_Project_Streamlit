#!/usr/bin/env python3
"""Diagnose the actual structure of the user's FEC file"""

import pandas as pd
import os

def diagnose_fec_file():
    """Check what's actually in the uploaded file"""
    
    # Look for CSV files in the directory that might be the uploaded file
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and '1900621' in f]
    
    if not csv_files:
        print("No matching CSV file found. Please upload your file again.")
        return
    
    file_path = csv_files[0]
    print(f"=== Analyzing file: {file_path} ===")
    
    try:
        # Read the file
        df = pd.read_csv(file_path)
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        
        print("\n=== All Column Names ===")
        for i, col in enumerate(df.columns):
            print(f"{i+1:2d}. '{col}'")
        
        print("\n=== Sample Data (First Row) ===")
        if len(df) > 0:
            first_row = df.iloc[0]
            for col in df.columns[:10]:  # Show first 10 columns
                value = str(first_row[col])[:50]  # Truncate long values
                print(f"{col}: {value}")
        
        print("\n=== Key Column Analysis ===")
        
        # Check for FORM TYPE values
        potential_form_cols = [col for col in df.columns if 'FORM' in col.upper() or col.upper() == 'A']
        if potential_form_cols:
            col = potential_form_cols[0]
            values = df[col].value_counts().head(5)
            print(f"FORM TYPE column '{col}' values: {dict(values)}")
        
        # Check for ENTITY TYPE values  
        potential_entity_cols = [col for col in df.columns if 'ENTITY' in col.upper() or 'TYPE' in col.upper() or col.upper() == 'F']
        if potential_entity_cols:
            col = potential_entity_cols[0]
            values = df[col].value_counts().head(5)
            print(f"ENTITY TYPE column '{col}' values: {dict(values)}")
            
        # Check for names
        name_cols = [col for col in df.columns if 'NAME' in col.upper()]
        print(f"Name-related columns: {name_cols}")
        
        # Check first few rows for SA values
        for col in df.columns:
            sample_values = df[col].dropna().astype(str).head(10).tolist()
            if any('SA' in str(val).upper() for val in sample_values):
                print(f"Column '{col}' contains SA values: {sample_values[:3]}")
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    diagnose_fec_file()