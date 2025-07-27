import requests
import pandas as pd
from config import FEC_API_KEY, FEC_BASE_URL

class FECAPIClient:
    def __init__(self):
        self.api_key = FEC_API_KEY
        self.base_url = FEC_BASE_URL
        self.session = requests.Session()
        self.session.params = {'api_key': self.api_key}
    
    def search_candidates(self, query, limit=20):
        """Search for candidates by name"""
        url = f"{self.base_url}/candidates/search/"
        params = {
            'q': query,
            'per_page': limit,
            'sort': 'name'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            candidates = []
            for candidate in data.get('results', []):
                candidates.append({
                    'candidate_id': candidate.get('candidate_id'),
                    'name': candidate.get('name'),
                    'party': candidate.get('party'),
                    'office': candidate.get('office'),
                    'state': candidate.get('state'),
                    'district': candidate.get('district'),
                    'cycles': candidate.get('cycles', [])
                })
            
            return candidates
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching candidates: {e}")
            return []
    
    def search_committees(self, query, limit=20):
        """Search for committees by name"""
        url = f"{self.base_url}/committees/"
        params = {
            'q': query,
            'per_page': limit,
            'sort': 'name'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            committees = []
            for committee in data.get('results', []):
                committees.append({
                    'committee_id': committee.get('committee_id'),
                    'name': committee.get('name'),
                    'committee_type': committee.get('committee_type'),
                    'designation': committee.get('designation'),
                    'party': committee.get('party'),
                    'state': committee.get('state'),
                    'cycles': committee.get('cycles', [])
                })
            
            return committees
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching committees: {e}")
            return []
    
    def get_candidate_committees(self, candidate_id, cycle=None):
        """Get committees associated with a candidate"""
        url = f"{self.base_url}/candidate/{candidate_id}/committees/"
        params = {}
        if cycle:
            params['cycle'] = cycle
            
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            committees = []
            for committee in data.get('results', []):
                committees.append({
                    'committee_id': committee.get('committee_id'),
                    'name': committee.get('name'),
                    'committee_type': committee.get('committee_type'),
                    'designation': committee.get('designation')
                })
            
            return committees
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting candidate committees: {e}")
            return []
    
    def get_schedule_a_data(self, committee_id=None, contributor_type=None, 
                          min_date=None, max_date=None, limit=1000):
        """
        Fetch Schedule A (receipts) data from FEC API
        
        Args:
            committee_id: Specific committee to fetch data for
            contributor_type: 'individual' or 'committee' 
            min_date: Start date (YYYY-MM-DD)
            max_date: End date (YYYY-MM-DD)
            limit: Maximum records to fetch
        """
        url = f"{self.base_url}/schedules/schedule_a/"
        params = {
            'per_page': min(limit, 100),  # API limit per page
            'sort': '-contribution_receipt_date'
        }
        
        if committee_id:
            params['committee_id'] = committee_id
        if contributor_type:
            params['contributor_type'] = contributor_type
        if min_date:
            params['min_date'] = min_date
        if max_date:
            params['max_date'] = max_date
        
        all_results = []
        page = 1
        
        try:
            while len(all_results) < limit:
                params['page'] = page
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                results = data.get('results', [])
                if not results:
                    break
                
                # Convert to our expected format (mimicking Excel columns)
                for result in results:
                    record = self.convert_api_to_excel_format(result)
                    all_results.append(record)
                
                # Check if we've reached the limit or end of data
                if len(results) < params['per_page'] or len(all_results) >= limit:
                    break
                
                page += 1
            
            return pd.DataFrame(all_results[:limit])
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Schedule A data: {e}")
            return pd.DataFrame()
    
    def convert_api_to_excel_format(self, api_result):
        """
        Convert FEC API result to Excel column format expected by processor
        Maps API fields to Excel columns (A, B, C, etc.)
        """
        # Determine contributor type for column F
        contributor_type = 'IND' if api_result.get('contributor_type') == 'individual' else 'COM'
        
        return {
            'A': 'SA',  # Schedule A indicator
            'F': contributor_type,  # Contributor type
            'G': api_result.get('contributor_name', ''),  # Vendor/Committee name
            'H': api_result.get('contributor_last_name', ''),  # Last name
            'I': api_result.get('contributor_first_name', ''),  # First name
            'P': api_result.get('contributor_state', ''),  # State
            'T': api_result.get('contribution_receipt_date', ''),  # Date
            'U': api_result.get('contribution_receipt_amount', 0),  # Amount
            'W': api_result.get('memo_text', ''),  # Memo
            'X': api_result.get('contributor_employer', ''),  # Employer
            'Z': api_result.get('contributor_name', ''),  # PAC name
            'committee_id': api_result.get('committee', {}).get('committee_id', ''),
            'filing_date': api_result.get('filing_date', ''),
            'transaction_id': api_result.get('transaction_id', '')
        }
    
    def get_committee_info(self, committee_id):
        """Get detailed information about a specific committee"""
        url = f"{self.base_url}/committee/{committee_id}/"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            results = data.get('results', [])
            if results:
                committee = results[0]
                return {
                    'committee_id': committee.get('committee_id'),
                    'name': committee.get('name'),
                    'committee_type': committee.get('committee_type'),
                    'committee_type_full': committee.get('committee_type_full'),
                    'designation': committee.get('designation'),
                    'designation_full': committee.get('designation_full'),
                    'party': committee.get('party'),
                    'party_full': committee.get('party_full'),
                    'state': committee.get('state'),
                    'cycles': committee.get('cycles', []),
                    'treasurer_name': committee.get('treasurer_name'),
                    'organization_type': committee.get('organization_type'),
                    'organization_type_full': committee.get('organization_type_full')
                }
            
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting committee info: {e}")
            return None