import pandas as pd
import sqlite3
import re
from difflib import SequenceMatcher
from functools import lru_cache
from database import get_connection

class FECDataProcessor:
    def __init__(self, database_config=None):
        self.conn = get_connection()
        # Store database configuration (default to all enabled if none provided)
        self.database_config = database_config or {
            'bad_donors': True,
            'rga_donors': True,
            'bad_employers': True
        }
        # Pre-compute normalized industry names for performance
        self.normalized_industry_names = self._precompute_industry_names()
        
        # FEC column name mappings to internal format
        self.fec_column_mapping = {
            'FORM TYPE': 'A',
            'ENTITY TYPE': 'F',
            'CONTRIBUTOR ORGANIZATION NAME': 'G',
            'CONTRIBUTOR LAST NAME': 'H',
            'CONTRIBUTOR FIRST NAME': 'I',
            'CONTRIBUTOR STATE': 'P',
            'CONTRIBUTION DATE': 'T',
            'CONTRIBUTION AMOUNT {F3L Bundled}': 'U',
            'CONTRIBUTION AMOUNT': 'U',  # Alternative name
            'MEMO TEXT/DESCRIPTION': 'W',
            'CONTRIBUTOR EMPLOYER': 'X',
            'DONOR COMMITTEE NAME': 'Z',
            'DONOR COMMITTEE FEC ID': 'committee_id'
        }
    
    def normalize_committee_name(self, name):
        """Normalize committee names for better fuzzy matching"""
        if not name or not name.strip():
            return ""
            
        name = str(name).upper().strip()
        
        # Common PAC variations - normalize to PAC
        name = re.sub(r'\bPOLITICAL ACTION COMMITTEE\b', 'PAC', name)
        name = re.sub(r'\bPOL ACTION COMMITTEE\b', 'PAC', name)
        name = re.sub(r'\bPOLITICAL ACTION COM\b', 'PAC', name)
        name = re.sub(r'\bPOL ACT COM\b', 'PAC', name)
        name = re.sub(r'\bPOLITICAL ACTION\b', 'PAC', name)
        
        # Common corporate abbreviations
        name = re.sub(r'\bCORPORATION\b', 'CORP', name)
        name = re.sub(r'\bINCORPORATED\b', 'INC', name)
        name = re.sub(r'\bLIMITED LIABILITY COMPANY\b', 'LLC', name)
        name = re.sub(r'\bLIMITED LIABILITY COM\b', 'LLC', name)
        name = re.sub(r'\bASSOCIATION\b', 'ASSOC', name)
        name = re.sub(r'\bCOMPANY\b', 'CO', name)
        name = re.sub(r'\bCOMMITTEE\b', 'COM', name)
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        return name
    
    def _precompute_industry_names(self):
        """Pre-compute normalized industry names for performance optimization"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT committee_name, larger_categories FROM industries WHERE committee_name IS NOT NULL AND committee_name != ""')
            industry_data = cursor.fetchall()
            
            normalized_cache = {}
            for committee_name, category in industry_data:
                if committee_name:
                    clean_name = committee_name.strip().upper()
                    normalized_name = self.normalize_committee_name(committee_name)
                    normalized_cache[committee_name] = {
                        'clean': clean_name,
                        'normalized': normalized_name,
                        'category': category
                    }
            
            return normalized_cache
        except Exception:
            # Return empty cache if database access fails
            return {}
    
    def detect_and_map_columns(self, df):
        """
        Detect if DataFrame uses FEC headers and map to internal format
        Returns: DataFrame with mapped columns
        """
        # Check if already using letter-based columns (Excel format)
        if 'A' in df.columns and 'F' in df.columns:
            return df  # Already in expected format
        
        # Create mapping for FEC format
        mapped_df = df.copy()
        
        # Map FEC column names to letter equivalents
        column_mapping = {}
        for fec_col, letter_col in self.fec_column_mapping.items():
            if fec_col in df.columns:
                column_mapping[fec_col] = letter_col
        
        # Rename columns
        mapped_df = mapped_df.rename(columns=column_mapping)
        
        # Handle CONTRIBUTOR ORGANIZATION NAME mapping to both G and Z
        if 'CONTRIBUTOR ORGANIZATION NAME' in df.columns:
            mapped_df['Z'] = df['CONTRIBUTOR ORGANIZATION NAME']  # For PAC name
            if 'G' not in mapped_df.columns:
                mapped_df['G'] = df['CONTRIBUTOR ORGANIZATION NAME']  # For vendor
        
        # Ensure committee_id is available if we have DONOR COMMITTEE FEC ID
        if 'DONOR COMMITTEE FEC ID' in df.columns:
            mapped_df['committee_id'] = df['DONOR COMMITTEE FEC ID']
        
        return mapped_df
    
    def process_fec_data(self, df):
        """
        Main processing function that mimics Excel macro logic
        Returns: dict with 'bad_donors' and 'pac_data' DataFrames
        """
        # First, map columns to expected format
        df = self.detect_and_map_columns(df)
        
        bad_donors = []
        pac_data = []
        
        for _, row in df.iterrows():
            # Check if it's a valid Schedule A transaction
            form_type = str(row.get('A', '')).strip().upper()
            if not (form_type.startswith('SA') or form_type == 'SA'):
                continue
                
            # Skip F3N summary records, only process SA records
            if form_type.startswith('F3'):
                continue
                
            # Individual donor classification (F = "IND")
            entity_type = str(row.get('F', '')).strip().upper()
            if entity_type == 'IND':
                donor_record = self.process_individual_donor(row)
                if donor_record:
                    bad_donors.append(donor_record)
            
            # Committee/PAC classification (F ≠ "IND" and G ≠ "WINRED")
            # Also handle cases where entity_type might be empty but we have organization name
            elif (entity_type not in ['IND', ''] and 
                  str(row.get('G', '')).upper() != 'WINRED') or \
                 (entity_type == '' and str(row.get('G', '')).strip() != ''):
                pac_record = self.process_committee_data(row)
                if pac_record:
                    pac_data.append(pac_record)
        
        # Convert to DataFrames
        bad_donors_df = pd.DataFrame(bad_donors) if bad_donors else pd.DataFrame()
        pac_data_df = pd.DataFrame(pac_data) if pac_data else pd.DataFrame()
        
        # Group duplicate donors before flagging
        if not bad_donors_df.empty:
            bad_donors_df = self.group_duplicate_donors(bad_donors_df)
            bad_donors_df = self.flag_individual_matches(bad_donors_df)
        
        if not pac_data_df.empty:
            pac_data_df = self.flag_committee_matches(pac_data_df)
        
        return {
            'bad_donors': bad_donors_df,
            'pac_data': pac_data_df
        }
    
    def process_individual_donor(self, row):
        """
        Process individual donor according to Excel macro mapping:
        First Name (Col I) -> A, Last Name (Col H) -> B, State (Col P) -> C,
        Employer (Col X) -> D, Memo (Col W) -> E, Date (Col T) -> F, Amount (Col U) -> G
        Also extract zip code for grouping purposes (usually Col Q)
        """
        first_name = str(row.get('I', '')).strip()
        last_name = str(row.get('H', '')).strip()
        state = str(row.get('P', '')).strip()
        zip_code = str(row.get('Q', '')).strip()  # Zip code for grouping
        employer = str(row.get('X', '')).strip()
        memo = str(row.get('W', '')).strip()
        date = str(row.get('T', '')).strip()
        amount = str(row.get('U', '')).strip()
        
        if not first_name or not last_name:
            return None
        
        # Generate match keys as specified
        full_key = f"{first_name.upper()} {last_name.upper()} {state.upper()}".strip()
        name_key = f"{first_name.upper()} {last_name.upper()}".strip()
        laststate_key = f"{last_name.upper()} {state.upper()}".strip()
        
        # Clean and convert amount
        try:
            amount_float = float(str(amount).replace('$', '').replace(',', '')) if amount else 0.0
        except:
            amount_float = 0.0
        
        return {
            'first_name': first_name,
            'last_name': last_name,
            'state': state,
            'zip_code': zip_code,
            'employer': employer,
            'memo': memo,
            'date': date,
            'amount': amount,
            'amount_numeric': amount_float,
            'full_key': full_key,
            'name_key': name_key,
            'laststate_key': laststate_key
        }
    
    def process_committee_data(self, row):
        """
        Process committee/PAC data according to correct mapping:
        PAC Name (Col G), Committee ID (Col Z), Memo (Col W), 
        Date (Col T), Amount (Col U)
        """
        pac_name = str(row.get('G', '')).strip()
        committee_id = str(row.get('Z', '')).strip()
        memo = str(row.get('W', '')).strip()
        date = str(row.get('T', '')).strip()
        amount = str(row.get('U', '')).strip()
        
        if not pac_name:
            return None
        
        return {
            'pac_name': pac_name,
            'committee_id': committee_id,
            'memo': memo,
            'date': date,
            'amount': amount
        }
    
    def flag_individual_matches(self, df):
        """Flag individual donors against relevant databases with confidence levels"""
        df = df.copy()
        df['flags'] = ''
        df['flag_sources'] = ''
        df['confidence_level'] = ''
        df['match_details'] = ''
        
        for idx, row in df.iterrows():
            flags = []
            sources = []
            confidence_info = []
            match_details = []
            
            # Check Bad Donors database with confidence levels
            if self.database_config.get('bad_donors', True):
                bad_donor_match = self.check_bad_donor_match(row)
                if bad_donor_match:
                    flags.append(f"BAD_DONOR_{bad_donor_match['confidence']}")
                    confidence_info.append(bad_donor_match['confidence'])
                    
                    # Create detailed source info
                    source_text = f"{bad_donor_match['color']} {bad_donor_match['confidence']} CONFIDENCE - {bad_donor_match['match_type']}: {bad_donor_match['affiliation']}"
                    if bad_donor_match.get('additional_matches', 0) > 0:
                        source_text += f" (+{bad_donor_match['additional_matches']} more matches)"
                    sources.append(source_text)
                    match_details.append(bad_donor_match)
            
            # Check RGA Donors databases (separate from bad donors but show in same section)
            if self.database_config.get('rga_donors', True):
                rga_donor_match = self.check_rga_donor_match(row)
                if rga_donor_match:
                    flags.append(f"BAD_DONOR_{rga_donor_match['confidence']}")
                    confidence_info.append(rga_donor_match['confidence'])
                    
                    # Create detailed source info
                    source_text = f"{rga_donor_match['color']} {rga_donor_match['confidence']} CONFIDENCE - {rga_donor_match['match_type']}: {rga_donor_match['affiliation']}"
                    if rga_donor_match.get('additional_matches', 0) > 0:
                        source_text += f" (+{rga_donor_match['additional_matches']} more matches)"
                    sources.append(source_text)
                    match_details.append(rga_donor_match)
            
            # Check Bad Employers database
            if self.database_config.get('bad_employers', True) and row['employer']:
                bad_employer_match = self.check_bad_employer_match(row['employer'])
                if bad_employer_match:
                    flags.append('BAD_EMPLOYER')
                    sources.append(f"🏢 Bad Employer: {bad_employer_match}")
            
            # Store results
            df.at[idx, 'flags'] = '|'.join(flags)
            df.at[idx, 'flag_sources'] = '|'.join(sources)
            df.at[idx, 'confidence_level'] = confidence_info[0] if confidence_info else ''
            df.at[idx, 'match_details'] = str(match_details[0]) if match_details else ''
            
            # Store RGA total information if available
            rga_match = next((detail for detail in match_details if 'RGA Donor' in detail.get('affiliation', '')), None)
            if rga_match:
                df.at[idx, 'rga_total'] = rga_match.get('rga_total', 0)
                df.at[idx, 'rga_contributions'] = rga_match.get('rga_contributions', 0)
        
        return df
    
    def group_duplicate_donors(self, df):
        """
        Group donors by name and zip code, combining their contributions
        """
        if df.empty:
            return df
        
        # Create grouping key based on name and zip code
        df['grouping_key'] = (df['first_name'].str.upper() + ' ' + 
                             df['last_name'].str.upper() + ' ' + 
                             df['zip_code'].str.strip()).str.strip()
        
        # Group by the key and aggregate data
        grouped_data = []
        
        for group_key, group_df in df.groupby('grouping_key'):
            if group_df.empty:
                continue
                
            # Use data from first record as base
            first_record = group_df.iloc[0].copy()
            
            # Aggregate amounts
            total_amount = group_df['amount_numeric'].sum()
            contribution_count = len(group_df)
            
            # Store individual dates and amounts for detailed display
            dates = group_df['date'].dropna().astype(str).tolist()
            amounts = group_df['amount_numeric'].tolist()
            
            # Create date range for backward compatibility
            if len(dates) > 1:
                date_range = f"{min(dates)} to {max(dates)}"
            elif len(dates) == 1:
                date_range = dates[0]
            else:
                date_range = ""
            
            # Combine memos (unique values only)
            memos = group_df['memo'].dropna().unique().tolist()
            combined_memo = " | ".join([memo for memo in memos if memo.strip()])
            
            # Combine employers (unique values only)
            employers = group_df['employer'].dropna().unique().tolist()
            combined_employer = " | ".join([emp for emp in employers if emp.strip()])
            
            # Create grouped record
            grouped_record = first_record.to_dict()
            grouped_record.update({
                'amount': f"${total_amount:,.2f}",
                'amount_numeric': total_amount,
                'date': date_range,
                'memo': combined_memo[:500] if combined_memo else first_record['memo'],  # Limit length
                'employer': combined_employer[:200] if combined_employer else first_record['employer'],
                'contribution_count': contribution_count,
                'is_grouped': contribution_count > 1,
                # Store individual contribution details
                'contribution_dates': dates,
                'contribution_amounts': amounts
            })
            
            # Remove the temporary grouping key
            if 'grouping_key' in grouped_record:
                del grouped_record['grouping_key']
            
            grouped_data.append(grouped_record)
        
        # Convert back to DataFrame
        result_df = pd.DataFrame(grouped_data)
        
        # Remove the temporary grouping key column if it exists
        if 'grouping_key' in result_df.columns:
            result_df = result_df.drop('grouping_key', axis=1)
        
        return result_df
    
    def flag_committee_matches(self, df):
        """Flag committee/PAC data against relevant databases"""
        df = df.copy()
        df['flags'] = ''
        df['flag_sources'] = ''
        
        for idx, row in df.iterrows():
            flags = []
            sources = []
            
            # Priority: Committee ID matching
            if row['committee_id']:
                # Check Bad Legislation by ID (moved from individual side)
                legislation_match = self.check_bad_legislation_by_id(row['committee_id'])
                if legislation_match:
                    flags.append('BAD_LEGISLATION')
                    sources.append(f"🗳️ Bad Legislation: {legislation_match}")
                
                # Check Bad Groups by ID
                bad_group_match = self.check_bad_group_by_id(row['committee_id'])
                if bad_group_match:
                    flags.append('BAD_GROUP')
                    sources.append(f"🚫 Bad Group (ID): {bad_group_match}")
                
                # Industry will be handled separately with Name → ID priority
                
                # Check LPAC by ID
                lpac_match = self.check_lpac_by_id(row['committee_id'])
                if lpac_match:
                    flags.append('LPAC')
                    sources.append(f"🏛️ LPAC: {lpac_match}")
            
            # Fallback: Name matching
            if row['pac_name']:
                # Check Bad Legislation by name (if no ID match)
                if not any('BAD_LEGISLATION' in f for f in flags):
                    legislation_match = self.check_bad_legislation_by_name(row['pac_name'])
                    if legislation_match:
                        flags.append('BAD_LEGISLATION')
                        sources.append(f"🗳️ Bad Legislation (Name): {legislation_match}")
                
                if not any('BAD_GROUP' in f for f in flags):
                    bad_group_match = self.check_bad_group_by_name(row['pac_name'])
                    if bad_group_match:
                        flags.append('BAD_GROUP')
                        sources.append(f"🚫 Bad Group (Name): {bad_group_match}")
                
                # Industry legacy name matching removed - handled separately below
            
            # Special handling for Industry: Name FIRST, then ID fallback
            industry_match = None
            industry_source = None
            
            # 1. Try fuzzy name matching first (primary)
            if row['pac_name']:
                industry_match = self.check_industry_by_name_fuzzy(row['pac_name'])
                if industry_match:
                    confidence_text = f" ({industry_match['confidence']}% match)"
                    industry_source = f"🏭 Industry (Name{confidence_text}): {industry_match['category']}"
            
            # 2. Fallback to exact Committee ID match only if no name match found
            if not industry_match and row['committee_id']:
                id_result = self.check_industry_by_id(row['committee_id'])
                if id_result:
                    industry_match = {'category': id_result}
                    industry_source = f"🏭 Industry (ID): {id_result}"
            
            # Add Industry flag if we found a match
            if industry_match:
                flags.append('INDUSTRY')
                sources.append(industry_source)
            
            df.at[idx, 'flags'] = '|'.join(flags)
            df.at[idx, 'flag_sources'] = '|'.join(sources)
        
        return df
    
    def check_bad_donor_match(self, row):
        """Check if donor matches Bad Donor database with confidence levels"""
        cursor = self.conn.cursor()
        matches = []
        
        # Level 1: High Confidence - Full Key (First + Last + State)
        cursor.execute('SELECT affiliation FROM bad_donors WHERE full_key = ?', (row['full_key'],))
        result = cursor.fetchone()
        if result:
            matches.append({
                'confidence': 'HIGH',
                'level': 1,
                'match_type': 'Full Name + State',
                'affiliation': result[0][:200],
                'color': '🔴'
            })
        
        # Level 2: Medium Confidence - Name Key (First + Last only)
        cursor.execute('SELECT affiliation FROM bad_donors WHERE name_key = ?', (row['name_key'],))
        results = cursor.fetchall()
        if results and not matches:  # Only if no high confidence match
            matches.append({
                'confidence': 'MEDIUM',
                'level': 2,
                'match_type': 'Full Name (any state)',
                'affiliation': results[0][0][:200],
                'color': '🟡',
                'additional_matches': len(results) - 1 if len(results) > 1 else 0
            })
        
        # Level 3: Low Confidence - EXCLUDED per user request
        
        return matches[0] if matches else None
    
    def check_bad_employer_match(self, employer):
        """Check if employer matches Bad Employer database"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT flag FROM bad_employers WHERE UPPER(name) = ?', (employer.upper(),))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def check_rga_donor_match(self, row):
        """Check if donor matches RGA Donors databases with ZIP code matching for high confidence"""
        cursor = self.conn.cursor()
        matches = []
        
        # Create ZIP code key from donor ZIP (first 5 digits)
        donor_zip = str(row.get('zip_code', '')).strip()[:5] if row.get('zip_code') else ''
        
        if donor_zip and len(donor_zip) == 5:
            # Level 1: HIGH Confidence - First + Last + ZIP match
            name_zip_key = f"{row['first_name'].upper()} {row['last_name'].upper()} {donor_zip}"
            
            # Check 2023 RGA donors
            cursor.execute('SELECT org_name, contrib_date, contribution_amt FROM rga_donors_2023 WHERE name_zip_key = ?', (name_zip_key,))
            results_2023 = cursor.fetchall()
            
            # Check 2024 RGA donors
            cursor.execute('SELECT org_name, contrib_date, contribution_amt FROM rga_donors_2024 WHERE name_zip_key = ?', (name_zip_key,))
            results_2024 = cursor.fetchall()
            
            all_results = results_2023 + results_2024
            
            if all_results:
                # Calculate total contributions and years
                total_amount = sum(float(result[2]) if result[2] else 0 for result in all_results)
                years = set()
                for result in all_results:
                    if result[1]:  # contrib_date
                        try:
                            year = result[1].split('/')[-1] if '/' in result[1] else result[1][:4]
                            years.add(year)
                        except:
                            pass
                
                years_text = ', '.join(sorted(years)) if years else 'Unknown'
                # Simplified affiliation text for cleaner display
                affiliation_text = f"{years_text} RGA Donor"
                
                matches.append({
                    'confidence': 'HIGH',
                    'level': 1,
                    'match_type': 'First + Last + ZIP',
                    'affiliation': affiliation_text,
                    'color': '🔴',
                    'additional_matches': len(all_results) - 1 if len(all_results) > 1 else 0,
                    # Store RGA total separately for dropdown display
                    'rga_total': total_amount,
                    'rga_contributions': len(all_results)
                })
        
        return matches[0] if matches else None
    
    
    def check_bad_group_by_id(self, committee_id):
        """Check Bad Groups by Committee ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT flag FROM bad_groups WHERE committee_id = ?', (committee_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def check_bad_group_by_name(self, committee_name):
        """Check Bad Groups by Committee Name"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT flag FROM bad_groups WHERE UPPER(committee_name) LIKE ?', 
                      (f'%{committee_name.upper()}%',))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def check_industry_by_id(self, committee_id):
        """Check Industry classification by Committee ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT larger_categories FROM industries WHERE committee_id = ?', (committee_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    @lru_cache(maxsize=1000)
    def _cached_fuzzy_match(self, committee_name_clean, committee_name_normalized, threshold=87):
        """Cached fuzzy matching helper using pre-computed normalized names for performance"""
        # Use pre-computed normalized industry names instead of database query
        if not self.normalized_industry_names:
            return None
        
        best_match = None
        best_score = 0
        best_category = None
        best_approach = None
        
        for db_name, data in self.normalized_industry_names.items():
            db_name_clean = data['clean']
            db_name_normalized = data['normalized']
            category = data['category']
            
            # Test 1: Original similarity
            original_similarity = SequenceMatcher(None, committee_name_clean, db_name_clean).ratio() * 100
            
            # Test 2: Normalized similarity (for PAC variations, etc.)
            normalized_similarity = SequenceMatcher(None, committee_name_normalized, db_name_normalized).ratio() * 100
            
            # Test 3: Containment logic on original names
            contains_score = 0
            if committee_name_clean in db_name_clean or db_name_clean in committee_name_clean:
                min_len = min(len(committee_name_clean), len(db_name_clean))
                max_len = max(len(committee_name_clean), len(db_name_clean))
                length_ratio = min_len / max_len
                contains_score = original_similarity + (length_ratio * 20)
            
            # Test 4: Containment logic on normalized names
            normalized_contains_score = 0
            if committee_name_normalized in db_name_normalized or db_name_normalized in committee_name_normalized:
                min_len = min(len(committee_name_normalized), len(db_name_normalized))
                max_len = max(len(committee_name_normalized), len(db_name_normalized))
                length_ratio = min_len / max_len
                normalized_contains_score = normalized_similarity + (length_ratio * 20)
            
            # Use the best score from all approaches
            final_score = max(original_similarity, normalized_similarity, contains_score, normalized_contains_score)
            
            # Track which approach worked best for debugging
            if final_score == normalized_similarity or final_score == normalized_contains_score:
                approach = "normalized"
            else:
                approach = "original"
            
            if final_score >= threshold and final_score > best_score:
                best_score = final_score
                best_match = db_name
                best_category = category
                best_approach = approach
        
        if best_match and best_score >= threshold:
            return {
                'category': best_category,
                'matched_name': best_match,
                'confidence': round(best_score, 1),
                'approach': best_approach
            }
        
        return None
    
    def check_industry_by_name_fuzzy(self, committee_name, threshold=87):
        """Check Industry classification by Committee Name using fuzzy matching with high certainty"""
        if not committee_name or not committee_name.strip():
            return None
            
        committee_name_clean = committee_name.strip().upper()
        committee_name_normalized = self.normalize_committee_name(committee_name)
        
        # Use cached helper function for performance
        return self._cached_fuzzy_match(committee_name_clean, committee_name_normalized, threshold)
    
    def check_industry_by_name(self, committee_name):
        """Check Industry classification by Committee Name (legacy exact/partial matching)"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT larger_categories FROM industries WHERE UPPER(committee_name) LIKE ?', 
                      (f'%{committee_name.upper()}%',))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def check_lpac_by_id(self, committee_id):
        """Check LPAC by Committee ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT pac_sponsor_district FROM lpac WHERE committee_id = ?', (committee_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def check_bad_legislation_by_id(self, committee_id):
        """Check Bad Legislation by Committee ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT associated_candidate FROM bad_legislation WHERE committee_id = ?', (committee_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def check_bad_legislation_by_name(self, committee_name):
        """Check Bad Legislation by Committee Name"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT associated_candidate FROM bad_legislation WHERE UPPER(committee_fec_name) LIKE ?', 
                      (f'%{committee_name.upper()}%',))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def get_industry_full_data_by_id(self, committee_id):
        """Get complete industry data by Committee ID for detailed display"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT larger_categories, org_type FROM industries WHERE committee_id = ?', (committee_id,))
        result = cursor.fetchone()
        if result:
            return {
                'larger_categories': result[0],
                'org_type': result[1]
            }
        return None
    
    def get_industry_full_data_by_name_fuzzy(self, committee_name, threshold=87):
        """Get complete industry data by name with fuzzy matching"""
        fuzzy_result = self.check_industry_by_name_fuzzy(committee_name, threshold)
        if fuzzy_result:
            # Get full data including org_type
            cursor = self.conn.cursor()
            cursor.execute('SELECT org_type FROM industries WHERE committee_name = ?', (fuzzy_result['matched_name'],))
            org_result = cursor.fetchone()
            
            fuzzy_result['org_type'] = org_result[0] if org_result else None
            return fuzzy_result
        return None
    
    def get_bad_group_full_data_by_id(self, committee_id):
        """Get complete bad group data by Committee ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT flag FROM bad_groups WHERE committee_id = ?', (committee_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def get_bad_group_full_data_by_name(self, committee_name):
        """Get complete bad group data by Committee Name"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT flag FROM bad_groups WHERE UPPER(committee_name) LIKE ?', 
                      (f'%{committee_name.upper()}%',))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def get_lpac_full_data_by_id(self, committee_id):
        """Get complete LPAC data by Committee ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT pac_sponsor_district FROM lpac WHERE committee_id = ?', (committee_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def batch_lookup_bad_donors(self, full_keys, name_keys):
        """Batch lookup bad donors for performance optimization"""
        cursor = self.conn.cursor()
        results = {}
        
        if full_keys:
            # Batch lookup full keys (high confidence)
            placeholders = ','.join(['?' for _ in full_keys])
            cursor.execute(f'SELECT full_key, affiliation FROM bad_donors WHERE full_key IN ({placeholders})', full_keys)
            for full_key, affiliation in cursor.fetchall():
                results[full_key] = {'confidence': 'HIGH', 'affiliation': affiliation[:200]}
        
        if name_keys:
            # Batch lookup name keys (medium confidence) for those without high confidence match
            remaining_name_keys = [nk for nk in name_keys if nk not in [fk.replace('_' + fk.split('_')[-1], '') for fk in results.keys()]]
            if remaining_name_keys:
                placeholders = ','.join(['?' for _ in remaining_name_keys])
                cursor.execute(f'SELECT name_key, affiliation FROM bad_donors WHERE name_key IN ({placeholders})', remaining_name_keys)
                for name_key, affiliation in cursor.fetchall():
                    if name_key not in [fk.replace('_' + fk.split('_')[-1], '') for fk in results.keys()]:
                        results[name_key] = {'confidence': 'MEDIUM', 'affiliation': affiliation[:200]}
        
        return results
    
    def batch_lookup_employers(self, employers):
        """Batch lookup bad employers for performance optimization"""
        if not employers:
            return {}
            
        cursor = self.conn.cursor()
        upper_employers = [emp.upper() for emp in employers]
        placeholders = ','.join(['?' for _ in upper_employers])
        cursor.execute(f'SELECT UPPER(name), flag FROM bad_employers WHERE UPPER(name) IN ({placeholders})', upper_employers)
        
        return {name: flag for name, flag in cursor.fetchall()}
    
    def batch_lookup_committees(self, committee_ids):
        """Batch lookup committee data from multiple tables for performance optimization"""
        if not committee_ids:
            return {}
            
        cursor = self.conn.cursor()
        results = {}
        placeholders = ','.join(['?' for _ in committee_ids])
        
        # Batch lookup bad groups
        cursor.execute(f'SELECT committee_id, flag FROM bad_groups WHERE committee_id IN ({placeholders})', committee_ids)
        for cid, flag in cursor.fetchall():
            if cid not in results:
                results[cid] = {}
            results[cid]['bad_group'] = flag
        
        # Batch lookup industries
        cursor.execute(f'SELECT committee_id, larger_categories FROM industries WHERE committee_id IN ({placeholders})', committee_ids)
        for cid, category in cursor.fetchall():
            if cid not in results:
                results[cid] = {}
            results[cid]['industry'] = category
            
        # Batch lookup LPAC
        cursor.execute(f'SELECT committee_id, pac_sponsor_district FROM lpac WHERE committee_id IN ({placeholders})', committee_ids)
        for cid, district in cursor.fetchall():
            if cid not in results:
                results[cid] = {}
            results[cid]['lpac'] = district
            
        # Batch lookup bad legislation
        cursor.execute(f'SELECT committee_id, associated_candidate FROM bad_legislation WHERE committee_id IN ({placeholders})', committee_ids)
        for cid, candidate in cursor.fetchall():
            if cid not in results:
                results[cid] = {}
            results[cid]['bad_legislation'] = candidate
        
        return results
    
    def extract_filer_info(self, df):
        """Extract filer state, district, and financial data from F3N row (row 2)"""
        try:
            # Look for F3N row in first 5 rows (F3N is always at the top)
            f3n_row = None
            search_rows = min(5, len(df))  # Only check first 5 rows for performance
            for i in range(search_rows):
                row = df.iloc[i]
                # Try column 'A' first (Excel format), then position 0 (CSV format)
                first_col_value = str(row.get('A', row.iloc[0] if len(row) > 0 else '')).strip().upper()
                if first_col_value.startswith('F3N'):
                    f3n_row = row
                    break
            
            if f3n_row is not None:
                # Determine if this is Excel format (letter columns) or CSV format (numeric positions)
                is_excel_format = 'J' in f3n_row.index if hasattr(f3n_row, 'index') else False
                
                if is_excel_format:
                    # Excel format - use letter column references
                    filer_state = str(f3n_row.get('J', '')).strip().upper()
                    filer_district = str(f3n_row.get('K', '')).strip()
                    committee_name = str(f3n_row.get('C', '')).strip()
                else:
                    # CSV format - use numeric positions (0-indexed)
                    # J = 10 (0-indexed = 9), K = 11 (0-indexed = 10), C = 3 (0-indexed = 2)
                    try:
                        filer_state = str(f3n_row.iloc[9] if len(f3n_row) > 9 else '').strip().upper()  # Column J
                        filer_district = str(f3n_row.iloc[10] if len(f3n_row) > 10 else '').strip()  # Column K
                        committee_name = str(f3n_row.iloc[2] if len(f3n_row) > 2 else '').strip()  # Column C
                    except (IndexError, KeyError):
                        filer_state = filer_district = committee_name = ''
                
                # Extract financial data with safe conversion to float
                def safe_float_convert(value):
                    try:
                        if pd.isna(value) or str(value).strip() == '':
                            return 0.0
                        return float(str(value).replace(',', '').replace('$', ''))
                    except (ValueError, TypeError):
                        return 0.0
                
                if is_excel_format:
                    # Excel format - use letter column references
                    receipts = safe_float_convert(f3n_row.get('BG', 0))
                    disbursements = safe_float_convert(f3n_row.get('BI', 0))
                    coh = safe_float_convert(f3n_row.get('BJ', 0))
                    debt = safe_float_convert(f3n_row.get('AF', 0))
                    loans_by_candidate = safe_float_convert(f3n_row.get('AO', 0))
                    small_donors_amount = safe_float_convert(f3n_row.get('AH', 0))
                else:
                    # CSV format - use numeric positions (0-indexed, +1 offset correction)
                    # User specified column mapping: AF=32, AH=34, AO=41, BG=59, BI=61, BJ=62
                    try:
                        debt = safe_float_convert(f3n_row.iloc[32] if len(f3n_row) > 32 else 0)  # AF = 32 (0-indexed = 32)
                        small_donors_amount = safe_float_convert(f3n_row.iloc[34] if len(f3n_row) > 34 else 0)  # AH = 34 (0-indexed = 34)
                        loans_by_candidate = safe_float_convert(f3n_row.iloc[41] if len(f3n_row) > 41 else 0)  # AO = 41 (0-indexed = 41)
                        receipts = safe_float_convert(f3n_row.iloc[59] if len(f3n_row) > 59 else 0)  # BG = 59 (0-indexed = 59)
                        disbursements = safe_float_convert(f3n_row.iloc[61] if len(f3n_row) > 61 else 0)  # BI = 61 (0-indexed = 61)
                        coh = safe_float_convert(f3n_row.iloc[62] if len(f3n_row) > 62 else 0)  # BJ = 62 (0-indexed = 62)
                    except (IndexError, KeyError):
                        # Fallback to zeros if positions don't exist
                        receipts = disbursements = coh = debt = loans_by_candidate = small_donors_amount = 0
                
                return {
                    'filer_state': filer_state if filer_state else None,
                    'filer_district': filer_district if filer_district else None,
                    'committee_name': committee_name if committee_name else None,
                    'financial_data': {
                        'receipts': receipts,
                        'disbursements': disbursements,
                        'coh': coh,
                        'debt': debt,
                        'loans_by_candidate': loans_by_candidate,
                        'small_donors_amount': small_donors_amount
                    }
                }
        except Exception as e:
            print(f"Error extracting filer info: {e}")
        
        return {
            'filer_state': None, 
            'filer_district': None,
            'committee_name': None,
            'financial_data': {
                'receipts': 0.0,
                'disbursements': 0.0,
                'coh': 0.0,
                'debt': 0.0,
                'loans_by_candidate': 0.0,
                'small_donors_amount': 0.0
            }
        }
    
    def calculate_derived_metrics(self, financial_data, bad_donors_df):
        """Calculate derived metrics from financial data and donor data"""
        try:
            # Extract financial values
            receipts = financial_data.get('receipts', 0)
            disbursements = financial_data.get('disbursements', 0)
            small_donors_amount = financial_data.get('small_donors_amount', 0)
            
            # Calculate burn rate
            burn_rate = 0
            if receipts > 0:
                burn_rate = (disbursements / receipts) * 100
            
            # Calculate % from small donors  
            small_donor_percentage = 0
            if receipts > 0:
                small_donor_percentage = (small_donors_amount / receipts) * 100
            
            # Calculate additional metrics from donor data
            # For total contributions, sum up contribution_count from all donors
            total_individual_contributions = 0
            if not bad_donors_df.empty and 'contribution_count' in bad_donors_df.columns:
                total_individual_contributions = bad_donors_df['contribution_count'].sum()
            else:
                total_individual_contributions = len(bad_donors_df) if not bad_donors_df.empty else 0
            
            # Calculate median individual contribution using individual amounts
            median_contribution = 0
            if not bad_donors_df.empty and 'contribution_amounts' in bad_donors_df.columns:
                # Flatten all individual contribution amounts
                all_amounts = []
                for amounts_list in bad_donors_df['contribution_amounts'].dropna():
                    if isinstance(amounts_list, list):
                        all_amounts.extend(amounts_list)
                if all_amounts:
                    median_contribution = pd.Series(all_amounts).median()
            elif not bad_donors_df.empty and 'amount_numeric' in bad_donors_df.columns:
                # Fallback to grouped amounts if individual amounts not available
                median_contribution = bad_donors_df['amount_numeric'].median()
            
            # Calculate max out donors using individual contribution amounts
            max_out_donors = 0
            if not bad_donors_df.empty and 'contribution_amounts' in bad_donors_df.columns:
                # Check individual contributions in the $3,400 - $3,600 range
                for amounts_list in bad_donors_df['contribution_amounts'].dropna():
                    if isinstance(amounts_list, list):
                        for amount in amounts_list:
                            if 3400 <= amount <= 3600:
                                max_out_donors += 1
            elif not bad_donors_df.empty and 'amount_numeric' in bad_donors_df.columns:
                # Fallback to grouped amounts (though this won't be as accurate)
                max_out_donors = len(bad_donors_df[
                    (bad_donors_df['amount_numeric'] >= 3400) & 
                    (bad_donors_df['amount_numeric'] <= 3600)
                ])
            
            return {
                'burn_rate': burn_rate,
                'small_donor_percentage': small_donor_percentage,
                'total_individual_contributions': total_individual_contributions,
                'median_contribution': median_contribution,
                'max_out_donors': max_out_donors
            }
            
        except Exception as e:
            print(f"Error calculating derived metrics: {e}")
            return {
                'burn_rate': 0,
                'small_donor_percentage': 0,
                'total_individual_contributions': 0,
                'median_contribution': 0,
                'max_out_donors': 0
            }
    
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
        """Get in-state vs out-of-state contribution breakdown"""
        if bad_donors_df.empty or not filer_state:
            return {
                'in_state_total': 0,
                'out_of_state_total': 0,
                'in_state_count': 0,
                'out_of_state_count': 0,
                'in_state_percentage': 0,
                'out_of_state_percentage': 0
            }
        
        # Filter for in-state vs out-of-state
        in_state = bad_donors_df[bad_donors_df['state'].str.upper() == filer_state.upper()]
        out_of_state = bad_donors_df[bad_donors_df['state'].str.upper() != filer_state.upper()]
        
        in_state_total = in_state['amount_numeric'].sum()
        out_of_state_total = out_of_state['amount_numeric'].sum()
        total_amount = in_state_total + out_of_state_total
        
        return {
            'in_state_total': in_state_total,
            'out_of_state_total': out_of_state_total,
            'in_state_count': len(in_state),
            'out_of_state_count': len(out_of_state),
            'in_state_percentage': (in_state_total / total_amount * 100) if total_amount > 0 else 0,
            'out_of_state_percentage': (out_of_state_total / total_amount * 100) if total_amount > 0 else 0
        }
    
    
    def close(self):
        """Close database connection safely"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except (sqlite3.ProgrammingError, AttributeError):
            # Ignore threading errors - connection will be cleaned up automatically
            pass