#!/usr/bin/env python3
"""
Database initialization script
Run this script to create and populate the SQLite database with your CSV files
"""

from database import create_database

if __name__ == "__main__":
    print("Initializing FEC Analysis Tool database...")
    create_database()
    print("Database initialization complete!")