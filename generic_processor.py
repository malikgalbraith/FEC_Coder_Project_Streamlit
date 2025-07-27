import pandas as pd
import sqlite3
import re
from difflib import SequenceMatcher
from database import get_connection

class GenericDataProcessor:
    def __init__(self):
        self.conn = get_connection()
    
    def process_generic_data(self, df, column_mapping):
        """
        Process generic CSV data with user-defined column mapping
        Assumes all donors are individual donors (no committee classification needed)
        """
        
        # Create mapped DataFrame with standardized column names
        mapped_df = self._map_columns(df, column_mapping)
        
        # Process individual donors only (as per user requirement)
        individual_donors = self._process_all_individual_donors(mapped_df)
        
        # Return results in the same format as FECDataProcessor
        results = {
            'bad_donors': individual_donors,
            'pac_data': pd.DataFrame(),  # Empty since all data treated as individual donors
            'summary': {
                'total_individual_donors': len(individual_donors),
                'total_committees': 0,
                'flagged_individuals': len(individual_donors[individual_donors['flags'] != '']) if not individual_donors.empty else 0,
                'flagged_committees': 0
            }
        }
        
        return results
    
    def _map_columns(self, df, column_mapping):
        """Map user columns to standardized internal format"""
        mapped_data = {}
        
        # Required fields
        mapped_data['first_name'] = df[column_mapping['first_name']] if column_mapping['first_name'] else ''
        mapped_data['last_name'] = df[column_mapping['last_name']] if column_mapping['last_name'] else ''
        mapped_data['state'] = df[column_mapping['state']] if column_mapping['state'] else ''
        mapped_data['amount'] = df[column_mapping['amount']] if column_mapping['amount'] else 0
        
        # Optional fields
        mapped_data['zip_code'] = df[column_mapping['zip_code']] if column_mapping['zip_code'] else ''
        mapped_data['date'] = df[column_mapping['date']] if column_mapping['date'] else ''
        mapped_data['employer'] = df[column_mapping['employer']] if column_mapping['employer'] else ''
        mapped_data['address'] = df[column_mapping['address']] if column_mapping['address'] else ''
        
        # Create DataFrame
        mapped_df = pd.DataFrame(mapped_data)
        
        # Clean and standardize data
        mapped_df['first_name'] = mapped_df['first_name'].astype(str).str.strip().str.upper()
        mapped_df['last_name'] = mapped_df['last_name'].astype(str).str.strip().str.upper()
        mapped_df['state'] = mapped_df['state'].astype(str).str.strip().str.upper()
        mapped_df['employer'] = mapped_df['employer'].astype(str).str.strip().str.upper()
        
        # Clean ZIP codes
        mapped_df['zip_code'] = mapped_df['zip_code'].astype(str).str.strip()
        mapped_df['zip_code'] = mapped_df['zip_code'].str.extract(r'(\d{5})')[0]  # Extract 5-digit ZIP
        
        # Enhanced amount processing for various currency formats
        mapped_df['amount'] = mapped_df['amount'].apply(self._parse_amount)
        
        return mapped_df
    
    def _parse_amount(self, amount_value):
        """Parse amount from various currency formats"""
        if pd.isna(amount_value) or amount_value == '':
            return 0.0
        
        try:
            # Convert to string and clean
            amount_str = str(amount_value).strip()
            
            # Handle common "no amount" indicators
            if amount_str.upper() in ['N/A', 'TBD', 'NULL', 'NONE', '-', '']:
                return 0.0
            
            # Remove currency symbols and formatting
            # Handle: $1,234.56, 1,234.56, $1234, 1234, etc.
            cleaned = amount_str.replace('$', '').replace(',', '').replace(' ', '')
            
            # Handle parentheses as negative (accounting format)
            is_negative = False
            if cleaned.startswith('(') and cleaned.endswith(')'):
                is_negative = True
                cleaned = cleaned[1:-1]  # Remove parentheses
            
            # Handle negative signs
            if cleaned.startswith('-'):
                is_negative = True
                cleaned = cleaned[1:]
            
            # Convert to float
            parsed_amount = float(cleaned)
            
            # Apply negative if needed
            if is_negative:
                parsed_amount = -parsed_amount
            
            return parsed_amount
            
        except (ValueError, TypeError, AttributeError) as e:
            print(f"DEBUG: Could not parse amount '{amount_value}': {e}")
            return 0.0
    
    def _process_all_individual_donors(self, df):
        """Process all individual donor data and flag against databases"""
        all_donors = []
        
        print(f"DEBUG: Starting to process {len(df)} rows")
        processed_count = 0
        skipped_count = 0
        
        for _, row in df.iterrows():
            # More flexible validation - require at least some identifier and state
            first_name = str(row['first_name']).strip() if not pd.isna(row['first_name']) else ''
            last_name = str(row['last_name']).strip() if not pd.isna(row['last_name']) else ''
            state = str(row['state']).strip() if not pd.isna(row['state']) else ''
            amount = row['amount'] if not pd.isna(row['amount']) else 0
            
            # Skip only if we have no name information AND no state, OR if amount is 0
            has_name_info = (first_name != '' or last_name != '')
            has_state = state != ''
            # Amount validation (already parsed by _parse_amount)
            has_valid_amount = isinstance(amount, (int, float)) and amount > 0
            
            if not has_name_info or not has_state or not has_valid_amount:
                skipped_count += 1
                if skipped_count <= 10:  # Show first 10 skipped rows for debugging
                    print(f"DEBUG: Skipping row - first_name: '{first_name}', last_name: '{last_name}', state: '{state}', amount: '{amount}' (has_name: {has_name_info}, has_state: {has_state}, valid_amount: {has_valid_amount})")
                continue
            
            processed_count += 1
            
            donor_data = {
                'first_name': first_name,
                'last_name': last_name,
                'state': state,
                'zip_code': row['zip_code'],
                'amount': row['amount'],
                'amount_numeric': row['amount'],  # Geographic analysis expects this column
                'date': row['date'],
                'employer': row['employer'],
                'address': row['address'],
                'memo': '',  # Not used in generic format
                'flags': '',
                'flag_sources': '',
                'confidence_level': '',
                'contribution_count': 1,
                'is_grouped': False
            }
            
            # Check against bad donor database
            bad_donor_flags = self._check_bad_donors(donor_data)
            
            # Check against bad employer database
            bad_employer_flags = self._check_bad_employers(donor_data)
            
            # Check against bad legislation database
            bad_legislation_flags = self._check_bad_legislation(donor_data)
            
            # Combine all flags
            all_flags = bad_donor_flags + bad_employer_flags + bad_legislation_flags
            
            # Set flag information (if any)
            if all_flags:
                donor_data['flags'] = '|'.join([f['flag'] for f in all_flags])
                donor_data['flag_sources'] = '|'.join([f['source'] for f in all_flags])
                donor_data['confidence_level'] = self._determine_confidence_level(all_flags)
            
            # Add ALL donors to the list (flagged and unflagged)
            all_donors.append(donor_data)
        
        print(f"DEBUG: Processing complete - {processed_count} rows processed, {skipped_count} rows skipped, {len(all_donors)} donors created")
        print(f"DEBUG: Processing rate: {(processed_count/(processed_count+skipped_count)*100):.1f}% of rows successfully processed")
        
        # Store processing statistics for UI display
        self.processing_stats = {
            'total_rows': len(df),
            'processed_count': processed_count,
            'skipped_count': skipped_count,
            'success_rate': (processed_count/(processed_count+skipped_count)*100) if (processed_count+skipped_count) > 0 else 0
        }
        
        return pd.DataFrame(all_donors)
    
    def get_processing_stats(self):
        """Get processing statistics for UI display"""
        return getattr(self, 'processing_stats', {
            'total_rows': 0,
            'processed_count': 0,
            'skipped_count': 0,
            'success_rate': 0
        })
    
    def _process_individual_donors(self, df):
        """Process individual donor data and flag against databases"""
        flagged_donors = []
        
        for _, row in df.iterrows():
            # Skip rows with missing required data
            if not row['first_name'] or not row['last_name'] or not row['state']:
                continue
            
            donor_data = {
                'first_name': row['first_name'],
                'last_name': row['last_name'],
                'state': row['state'],
                'zip_code': row['zip_code'],
                'amount': row['amount'],
                'amount_numeric': row['amount'],  # Geographic analysis expects this column
                'date': row['date'],
                'employer': row['employer'],
                'address': row['address'],
                'memo': '',  # Not used in generic format
                'flags': '',
                'flag_sources': '',
                'confidence_level': '',
                'contribution_count': 1,
                'is_grouped': False
            }
            
            # Check against bad donor database
            bad_donor_flags = self._check_bad_donors(donor_data)
            
            # Check against bad employer database
            bad_employer_flags = self._check_bad_employers(donor_data)
            
            # Check against bad legislation database
            bad_legislation_flags = self._check_bad_legislation(donor_data)
            
            # Combine all flags
            all_flags = bad_donor_flags + bad_employer_flags + bad_legislation_flags
            
            if all_flags:
                donor_data['flags'] = '|'.join([f['flag'] for f in all_flags])
                donor_data['flag_sources'] = '|'.join([f['source'] for f in all_flags])
                donor_data['confidence_level'] = self._determine_confidence_level(all_flags)
                flagged_donors.append(donor_data)
        
        return pd.DataFrame(flagged_donors)
    
    def _check_bad_donors(self, donor_data):
        """Check donor against bad donors database"""
        flags = []
        
        try:
            cursor = self.conn.cursor()
            
            # Create matching keys
            full_key = f"{donor_data['first_name']} {donor_data['last_name']} {donor_data['state']}"
            name_key = f"{donor_data['first_name']} {donor_data['last_name']}"
            last_state_key = f"{donor_data['last_name']} {donor_data['state']}"
            
            # Check High Confidence - Full Key (First + Last + State)
            cursor.execute('SELECT affiliation FROM bad_donors WHERE full_key = ?', (full_key,))
            result = cursor.fetchone()
            if result:
                print(f"DEBUG: Found HIGH confidence bad donor match for {name_key}")
                flags.append({
                    'flag': 'BAD_DONOR',
                    'source': result[0][:200],  # Truncate long descriptions
                    'confidence': 'HIGH'
                })
            
            # Check Medium Confidence - Name Key (First + Last only) if no high confidence match
            if not flags:
                cursor.execute('SELECT affiliation FROM bad_donors WHERE name_key = ?', (name_key,))
                result = cursor.fetchone()
                if result:
                    print(f"DEBUG: Found MEDIUM confidence bad donor match for {name_key}")
                    flags.append({
                        'flag': 'BAD_DONOR',
                        'source': result[0][:200],  # Truncate long descriptions
                        'confidence': 'MEDIUM'
                    })
        
        except Exception as e:
            print(f"Error checking bad donors: {e}")
        
        return flags
    
    def _check_bad_employers(self, donor_data):
        """Check employer against bad employers database"""
        flags = []
        
        if not donor_data['employer'] or str(donor_data['employer']).strip() == '':
            return flags
        
        try:
            cursor = self.conn.cursor()
            
            # Check exact match
            cursor.execute("""
                SELECT DISTINCT name, flag 
                FROM bad_employers 
                WHERE UPPER(name) = ?
            """, (donor_data['employer'],))
            
            results = cursor.fetchall()
            if results:
                print(f"DEBUG: Found bad employer match for {donor_data['employer']}: {len(results)} matches")
            
            for employer_name, flag_desc in results:
                flags.append({
                    'flag': 'BAD_EMPLOYER',
                    'source': flag_desc,
                    'confidence': 'HIGH'
                })
        
        except Exception as e:
            print(f"Error checking bad employers: {e}")
        
        return flags
    
    def _check_bad_legislation(self, donor_data):
        """Check donor against bad legislation database - not applicable for individual donors"""
        # Bad legislation analysis is for PACs/committees, not individual donors
        # For generic data (all treated as individual donors), return empty flags
        return []
    
    def _determine_confidence_level(self, flags):
        """Determine overall confidence level from flags"""
        if any(f['confidence'] == 'HIGH' for f in flags):
            return 'HIGH'
        elif any(f['confidence'] == 'MEDIUM' for f in flags):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def group_donors_by_name_state(self, donors_df):
        """Group donors by name and state, summing amounts"""
        if donors_df.empty:
            return donors_df
        
        # Group by first_name, last_name, state
        grouped = donors_df.groupby(['first_name', 'last_name', 'state']).agg({
            'amount': 'sum',
            'zip_code': 'first',
            'date': lambda x: ', '.join(x.dropna().astype(str)),
            'employer': 'first',
            'address': 'first',
            'memo': lambda x: ', '.join(x.dropna().astype(str)),
            'flags': 'first',
            'flag_sources': 'first',
            'confidence_level': 'first'
        }).reset_index()
        
        # Add contribution count
        contribution_counts = donors_df.groupby(['first_name', 'last_name', 'state']).size()
        grouped['contribution_count'] = grouped.apply(
            lambda row: contribution_counts[(row['first_name'], row['last_name'], row['state'])], 
            axis=1
        )
        
        # Mark as grouped if more than one contribution
        grouped['is_grouped'] = grouped['contribution_count'] > 1
        
        return grouped
    
    def get_geographic_analysis(self, bad_donors_df):
        """Analyze geographic distribution of individual donors"""
        if bad_donors_df.empty:
            return {
                'state_totals': pd.DataFrame(),
                'zip_data': pd.DataFrame(),
                'summary_stats': {}
            }
        
        # Aggregate by state
        state_totals = bad_donors_df.groupby('state').agg({
            'amount_numeric': 'sum',
            'first_name': 'count'  # Count of contributions
        }).reset_index()
        state_totals.columns = ['state', 'total_amount', 'contributor_count']
        state_totals = state_totals.sort_values('total_amount', ascending=False)
        
        # Calculate percentages
        total_amount = state_totals['total_amount'].sum()
        state_totals['percentage'] = (state_totals['total_amount'] / total_amount * 100).round(2)
        
        # Aggregate by zip code (for detailed mapping)
        zip_data = bad_donors_df.groupby(['state', 'zip_code']).agg({
            'amount_numeric': 'sum',
            'first_name': 'count'
        }).reset_index()
        zip_data.columns = ['state', 'zip_code', 'total_amount', 'contributor_count']
        
        # Summary statistics
        total_contributors = len(bad_donors_df)
        total_amount = bad_donors_df['amount_numeric'].sum()
        unique_states = bad_donors_df['state'].nunique()
        
        summary_stats = {
            'total_contributors': total_contributors,
            'total_amount': total_amount,
            'unique_states': unique_states,
            'average_contribution': total_amount / total_contributors if total_contributors > 0 else 0
        }
        
        return {
            'state_totals': state_totals,
            'zip_data': zip_data,
            'summary_stats': summary_stats
        }
    
    def get_state_breakdown(self, bad_donors_df, filer_state):
        """Get in-state vs out-of-state breakdown"""
        if bad_donors_df.empty or not filer_state:
            return None
        
        # For generic data, we don't have a filer state, so return None
        # This will cause the display to show "No breakdown data available"
        return None
    
    def extract_filer_info(self, df):
        """Extract filer information - not applicable for generic data"""
        # Generic CSV data doesn't have filer information
        return {'filer_state': None, 'filer_district': None}
    
    def close(self):
        """Close database connection safely"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except (sqlite3.ProgrammingError, AttributeError):
            # Ignore threading errors - connection will be cleaned up automatically
            pass