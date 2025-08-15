import streamlit as st
import pandas as pd
import os
import csv
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
from database import create_database, get_connection
from data_processor import FECDataProcessor
from fec_api import FECAPIClient

# Page configuration
st.set_page_config(
    page_title="FEC Donor & PAC Analysis Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global ZIP coordinate cache
ZIP_COORDINATES_CACHE = None

def load_zip_coordinates():
    """Load ZIP coordinates from comprehensive CSV database"""
    global ZIP_COORDINATES_CACHE
    
    if ZIP_COORDINATES_CACHE is not None:
        return ZIP_COORDINATES_CACHE
    
    zip_coords = {}
    csv_file = 'Databases/georef-united-states-of-america-zc-point.csv'
    
    try:
        if not os.path.exists(csv_file):
            st.warning(f"ZIP coordinate database not found: {csv_file}")
            return {}
        
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)  # Skip header row
            
            for row in reader:
                if len(row) >= 17:
                    zip_code = row[0].strip()
                    geo_point = row[16].strip()
                    
                    # Parse coordinate format: "lat, lon"
                    if geo_point and ',' in geo_point:
                        try:
                            lat_str, lon_str = geo_point.split(',', 1)
                            lat = float(lat_str.strip())
                            lon = float(lon_str.strip())
                            zip_coords[zip_code] = (lat, lon)
                        except (ValueError, IndexError):
                            # Skip malformed coordinates
                            continue
        
        ZIP_COORDINATES_CACHE = zip_coords
        return zip_coords
        
    except Exception as e:
        st.error(f"Error loading ZIP coordinate database: {e}")
        return {}

def load_major_cities(state_filter=None, population_threshold=50000, max_cities=30):
    """Load major cities from ZIP coordinate database for map labeling"""
    csv_file = 'Databases/georef-united-states-of-america-zc-point.csv'
    major_cities = []
    
    try:
        if not os.path.exists(csv_file):
            return []
        
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)  # Skip header row
            
            for row in reader:
                if len(row) >= 17:
                    zip_code = row[0].strip()
                    city_name = row[1].strip()
                    state_code = row[2].strip()
                    population_str = row[6].strip()
                    geo_point = row[16].strip()
                    
                    # Filter by state if specified
                    if state_filter and state_code.upper() != state_filter.upper():
                        continue
                    
                    # Parse population
                    try:
                        population = float(population_str) if population_str else 0
                    except (ValueError, TypeError):
                        population = 0
                    
                    # Filter by population threshold
                    if population < population_threshold:
                        continue
                    
                    # Parse coordinates
                    if geo_point and ',' in geo_point:
                        try:
                            lat_str, lon_str = geo_point.split(',', 1)
                            lat = float(lat_str.strip())
                            lon = float(lon_str.strip())
                            
                            major_cities.append({
                                'city': city_name,
                                'state': state_code,
                                'population': population,
                                'lat': lat,
                                'lon': lon,
                                'zip_code': zip_code
                            })
                        except (ValueError, IndexError):
                            continue
        
        # Sort by population (descending) and limit to max_cities
        major_cities.sort(key=lambda x: x['population'], reverse=True)
        return major_cities[:max_cities]
        
    except Exception as e:
        st.error(f"Error loading major cities: {e}")
        return []

def get_viridis_color(amount, min_amount, max_amount):
    """Convert contribution amount to Viridis color scale (purple→blue→green→yellow)"""
    if max_amount == min_amount:
        return '#440154'  # Default purple for single value
    
    # Normalize amount to 0-1 range
    normalized = (amount - min_amount) / (max_amount - min_amount)
    
    # Viridis color scale points (5 key colors)
    viridis_colors = [
        '#440154',  # Dark purple (low values)
        '#3b518b',  # Blue
        '#21908c',  # Teal
        '#f97306',  # Bright orange  
        '#e31a1c'   # Strong red (high values)
    ]
    
    # Find which segment the normalized value falls into
    if normalized <= 0:
        return viridis_colors[0]
    elif normalized >= 1:
        return viridis_colors[-1]
    else:
        # Scale to 0-4 range and get color index
        scaled = normalized * (len(viridis_colors) - 1)
        lower_idx = int(scaled)
        upper_idx = min(lower_idx + 1, len(viridis_colors) - 1)
        
        # If exactly on a color point, return that color
        if lower_idx == upper_idx:
            return viridis_colors[lower_idx]
        
        # Otherwise interpolate between two colors
        ratio = scaled - lower_idx
        lower_color = viridis_colors[lower_idx]
        upper_color = viridis_colors[upper_idx]
        
        # Simple color interpolation (hex to RGB to hex)
        def hex_to_rgb(hex_color):
            return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        
        def rgb_to_hex(rgb):
            return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
        
        lower_rgb = hex_to_rgb(lower_color)
        upper_rgb = hex_to_rgb(upper_color)
        
        interpolated_rgb = tuple(
            lower_rgb[i] + ratio * (upper_rgb[i] - lower_rgb[i])
            for i in range(3)
        )
        
        return rgb_to_hex(interpolated_rgb)

def get_available_states_for_dropdown(geographic_data):
    """Extract and format available states from geographic data for dropdown"""
    if geographic_data is None or 'state_totals' not in geographic_data:
        return []
    
    state_totals = geographic_data['state_totals']
    if state_totals.empty:
        return []
    
    # Sort by total amount descending to show most active states first
    sorted_states = state_totals.sort_values('total_amount', ascending=False)
    
    # Format as "STATE - $amount (count contributors)"
    formatted_states = []
    for _, row in sorted_states.iterrows():
        state_code = row['state']
        amount = row['total_amount']
        count = row['contributor_count']
        formatted_states.append(f"{state_code} - ${amount:,.0f} ({count:,} contributors)")
    
    return formatted_states

def initialize_database():
    """Initialize database if it doesn't exist"""
    if not os.path.exists("fec_master.db"):
        with st.spinner("Setting up database for first time..."):
            create_database()
        st.success("Database initialized successfully!")

def get_database_config():
    """Get current database configuration from session state"""
    return st.session_state.get('database_config', {
        'bad_donors': True,
        'rga_donors': True, 
        'bad_employers': True
    })

def is_database_enabled(db_name):
    """Check if a specific database is enabled for analysis"""
    config = get_database_config()
    return config.get(db_name, True)

def sort_pac_data_by_priority(pac_data_df):
    """Sort PAC/committee data by flag priority and amount"""
    if pac_data_df.empty:
        return pac_data_df
    
    def get_pac_priority_score(row):
        """Assign priority scores for PAC sorting"""
        flags = row.get('flags', '')
        
        # Bad Groups get highest priority (score 1)
        if 'BAD_GROUP' in flags:
            return 1
        # Industry classifications (score 2)
        elif 'INDUSTRY' in flags:
            return 2
        # LPAC matches (score 3)
        elif 'LPAC' in flags:
            return 3
        # Bad Legislation (score 4) - will be filtered out anyway
        elif 'BAD_LEGISLATION' in flags:
            return 4
        else:
            return 5  # Other flags
    
    # Add priority score and sort
    pac_data_df = pac_data_df.copy()
    pac_data_df['priority_score'] = pac_data_df.apply(get_pac_priority_score, axis=1)
    
    # Convert amount to numeric for sorting
    pac_data_df['amount_numeric'] = pd.to_numeric(
        pac_data_df['amount'].str.replace('$', '').str.replace(',', ''), 
        errors='coerce'
    ).fillna(0)
    
    # Sort by priority score, then by amount (highest first)
    pac_data_df = pac_data_df.sort_values(['priority_score', 'amount_numeric'], ascending=[True, False])
    
    return pac_data_df.drop('priority_score', axis=1)

def sort_donors_by_priority(flagged_donors):
    """Sort flagged donors by priority: Bad Employer > High > Medium > Low confidence"""
    if flagged_donors.empty:
        return flagged_donors
    
    def get_priority_score(row):
        """Assign priority scores for sorting"""
        flags = row.get('flags', '')
        confidence = row.get('confidence_level', '')
        
        # Bad Employer gets highest priority (score 1)
        if 'BAD_EMPLOYER' in flags:
            return 1
        # Then by confidence level
        elif confidence == 'HIGH':
            return 2
        elif confidence == 'MEDIUM':
            return 3
        else:
            return 5  # Other flags
    
    # Add priority score and sort
    flagged_donors = flagged_donors.copy()
    flagged_donors['priority_score'] = flagged_donors.apply(get_priority_score, axis=1)
    
    # Sort by priority score, then by amount (highest first)
    flagged_donors['amount_numeric'] = flagged_donors['amount_numeric'].fillna(0)
    flagged_donors = flagged_donors.sort_values(['priority_score', 'amount_numeric'], ascending=[True, False])
    
    return flagged_donors.drop('priority_score', axis=1)

def clean_flag_description(flag_sources):
    """Clean up flag descriptions for email report"""
    if not flag_sources:
        return "Unknown Flag"
    
    # Extract meaningful descriptions
    sources = flag_sources.split('|')
    clean_descriptions = []
    
    for source in sources:
        if 'HIGH CONFIDENCE' in source and 'Insurrectionist' in source:
            clean_descriptions.append("Jan 6th Participant")
        elif 'MEDIUM CONFIDENCE' in source and 'Insurrectionist' in source:
            clean_descriptions.append("Jan 6th Participant (verify location)")
        elif 'LOW CONFIDENCE' in source and 'Insurrectionist' in source:
            clean_descriptions.append("Jan 6th Participant (verify identity)")
        elif 'Bad Employer' in source:
            # Extract employer name if available
            employer_part = source.split('Bad Employer: ')[-1]
            clean_descriptions.append(f"Bad Employer: {employer_part}")
        elif 'RGA Donor' in source:
            # Use the simplified RGA format directly
            clean_descriptions.append(source)
        elif 'HIGH CONFIDENCE' in source:
            # Extract affiliation info
            affiliation = source.split(': ')[-1] if ': ' in source else source
            clean_descriptions.append(affiliation[:50] + "..." if len(affiliation) > 50 else affiliation)
        else:
            # Fallback for other types
            clean_descriptions.append(source[:50] + "..." if len(source) > 50 else source)
    
    return " | ".join(clean_descriptions)

def generate_email_report(flagged_donors):
    """Generate email-ready report for high confidence matches"""
    if flagged_donors.empty:
        return ""
    
    # Filter to high confidence and bad employer matches
    high_priority = flagged_donors[
        (flagged_donors['confidence_level'] == 'HIGH') | 
        (flagged_donors['flags'].str.contains('BAD_EMPLOYER', na=False))
    ]
    
    if high_priority.empty:
        return ""
    
    report_lines = ["High confidence donor matches:\n"]
    total_amount = 0
    
    for _, donor in high_priority.iterrows():
        first_name = donor['first_name']
        last_name = donor['last_name']
        amount = donor.get('amount', '$0.00')
        amount_numeric = donor.get('amount_numeric', 0)
        
        # Get specific flag description based on flag sources and match details
        flags = donor.get('flags', '')
        flag_sources = donor.get('flag_sources', '')
        match_details = donor.get('match_details', [])
        
        flag_desc = "Flagged"  # Default fallback
        
        # Handle Bad Employer first (highest priority)
        if 'BAD_EMPLOYER' in flags:
            flag_desc = "Bad Employer"
        
        # Handle RGA donors 
        elif 'RGA Donor' in flag_sources:
            flag_desc = "RGA donor"
            
        # Handle Bad Donor Master - extract actual affiliation
        elif 'BAD_DONOR' in flags:
            # Try to get affiliation from match_details first (FEC processor)
            if match_details:
                try:
                    # Parse match_details string to extract affiliation
                    detail_str = str(match_details)
                    if 'affiliation' in detail_str and 'Insurrectionist' in detail_str:
                        flag_desc = "Insurrectionist"
                    elif 'affiliation' in detail_str:
                        # Extract affiliation text between quotes
                        import re
                        affiliation_match = re.search(r"'affiliation': '([^']*)'", detail_str)
                        if affiliation_match:
                            full_affiliation = affiliation_match.group(1)
                            # Simplify long affiliations for email
                            if 'Insurrectionist' in full_affiliation:
                                flag_desc = "Insurrectionist"
                            else:
                                # Use first meaningful part
                                flag_desc = full_affiliation.split(' charged with')[0] if ' charged with' in full_affiliation else full_affiliation[:50]
                except:
                    flag_desc = "Bad Donor"
            
            # Try to get affiliation from flag_sources (Generic processor)
            elif flag_sources and flag_sources != "Bad Donor":
                if 'Insurrectionist' in flag_sources:
                    flag_desc = "Insurrectionist"
                else:
                    # Extract meaningful part from flag sources
                    flag_desc = flag_sources.split(' - ')[-1] if ' - ' in flag_sources else flag_sources[:50]
            else:
                flag_desc = "Bad Donor"
        
        # Simplified format: First Last, Total Amount, Flag
        line = f"• {first_name} {last_name}, {amount}, {flag_desc}"
        report_lines.append(line)
        
        total_amount += amount_numeric
    
    # Add summary
    report_lines.append(f"\nTotal amount: ${total_amount:,.2f}")
    report_lines.append(f"Total count: {len(high_priority)} donors")
    
    return "\n".join(report_lines)

def generate_fec_financial_email_report():
    """Generate email-ready financial summary for FEC reports"""
    if ('fec_financial_data' not in st.session_state or 
        'fec_derived_metrics' not in st.session_state or
        'fec_committee_name' not in st.session_state):
        return ""
    
    financial_data = st.session_state['fec_financial_data']
    derived_metrics = st.session_state['fec_derived_metrics']
    committee_name = st.session_state['fec_committee_name']
    
    # Get state and district info
    filer_state = ""
    filer_district = ""
    if 'current_processor' in st.session_state and 'original_df' in st.session_state:
        current_processor = st.session_state['current_processor']
        original_df = st.session_state['original_df']
        filer_info = current_processor.extract_filer_info(original_df)
        filer_state = filer_info.get('filer_state', '')
        filer_district = filer_info.get('filer_district', '')
    
    # Build email report
    report_lines = []
    
    # Header with bold committee name
    if filer_state and filer_district and committee_name:
        report_lines.append(f"**{filer_state}-{filer_district} {committee_name}**\n")
    elif committee_name:
        report_lines.append(f"**{committee_name}**\n")
    
    # Financial metrics
    report_lines.append(f"Receipts: ${financial_data.get('receipts', 0):,.2f}")
    report_lines.append(f"Disbursements: ${financial_data.get('disbursements', 0):,.2f}")
    report_lines.append(f"COH: ${financial_data.get('coh', 0):,.2f}")
    report_lines.append(f"Debt: ${financial_data.get('debt', 0):,.2f}")
    
    # Only include loans if amount > 0
    loans = financial_data.get('loans_by_candidate', 0)
    if loans > 0:
        report_lines.append(f"Loans by Candidate: ${loans:,.2f}")
    
    # Derived metrics
    burn_rate = derived_metrics.get('burn_rate', 0)
    small_donor_pct = derived_metrics.get('small_donor_percentage', 0)
    
    report_lines.append(f"Burn Rate: {burn_rate:.1f}%")
    report_lines.append(f"% from Small Donors: {small_donor_pct:.1f}%")
    
    # Additional metrics
    total_contributions = derived_metrics.get('total_individual_contributions', 0)
    median_contribution = derived_metrics.get('median_contribution', 0)
    max_out_donors = derived_metrics.get('max_out_donors', 0)
    
    report_lines.append(f"Total Individual Contributions: {total_contributions:,}")
    report_lines.append(f"Median Individual Contribution: ${median_contribution:.2f}")
    report_lines.append(f"Max Out Donors: {max_out_donors:,}")
    
    return "\n".join(report_lines)

def display_dashboard_cards(flagged_donors):
    """Display dashboard summary cards for different priority levels"""
    if flagged_donors.empty:
        st.info("No flagged donors found.")
        return
    
    # Group donors by priority
    bad_employer = flagged_donors[flagged_donors['flags'].str.contains('BAD_EMPLOYER', na=False)]
    high_conf = flagged_donors[flagged_donors['confidence_level'] == 'HIGH']
    medium_conf = flagged_donors[flagged_donors['confidence_level'] == 'MEDIUM']
    
    # Calculate totals
    def calc_totals(df):
        if df.empty:
            return 0, 0.0
        count = len(df)
        total = df['amount_numeric'].sum()
        return count, total
    
    bad_emp_count, bad_emp_total = calc_totals(bad_employer)
    high_count, high_total = calc_totals(high_conf)
    medium_count, medium_total = calc_totals(medium_conf)
    
    # Create dashboard cards
    st.subheader("🎯 Investigation Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if bad_emp_count > 0:
            st.markdown(f"""
            <div style="border: 2px solid #ff4b4b; border-radius: 10px; padding: 20px; text-align: center; background-color: #fff5f5;">
                <h3 style="color: #ff4b4b; margin: 0;">🏢 Bad Employer</h3>
                <h2 style="margin: 10px 0;">{bad_emp_count} matches</h2>
                <p style="margin: 5px 0; font-size: 18px;">${bad_emp_total:,.2f} total</p>
                <p style="margin: 0; color: #666; font-size: 14px;">Systemic corruption - highest priority</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View Bad Employer Details", key="bad_emp", use_container_width=True):
                st.session_state['show_section'] = 'bad_employer'
        else:
            st.markdown("""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center; background-color: #f9f9f9;">
                <h3 style="color: #999; margin: 0;">🏢 Bad Employer</h3>
                <h2 style="margin: 10px 0; color: #999;">0 matches</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if high_count > 0:
            st.markdown(f"""
            <div style="border: 2px solid #ff6b6b; border-radius: 10px; padding: 20px; text-align: center; background-color: #fff5f5;">
                <h3 style="color: #ff6b6b; margin: 0;">🔴 High Confidence</h3>
                <h2 style="margin: 10px 0;">{high_count} matches</h2>
                <p style="margin: 5px 0; font-size: 18px;">${high_total:,.2f} total</p>
                <p style="margin: 0; color: #666; font-size: 14px;">Exact name + state matches</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View High Confidence Details", key="high_conf", use_container_width=True):
                st.session_state['show_section'] = 'high_confidence'
        else:
            st.markdown("""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center; background-color: #f9f9f9;">
                <h3 style="color: #999; margin: 0;">🔴 High Confidence</h3>
                <h2 style="margin: 10px 0; color: #999;">0 matches</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if medium_count > 0:
            st.markdown(f"""
            <div style="border: 2px solid #ffa726; border-radius: 10px; padding: 20px; text-align: center; background-color: #fff8f0;">
                <h3 style="color: #ffa726; margin: 0;">🟡 Medium Confidence</h3>
                <h2 style="margin: 10px 0;">{medium_count} matches</h2>
                <p style="margin: 5px 0; font-size: 18px;">${medium_total:,.2f} total</p>
                <p style="margin: 0; color: #666; font-size: 14px;">Name matches - verify location</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View Medium Confidence Details", key="medium_conf", use_container_width=True):
                st.session_state['show_section'] = 'medium_confidence'
        else:
            st.markdown("""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center; background-color: #f9f9f9;">
                <h3 style="color: #999; margin: 0;">🟡 Medium Confidence</h3>
                <h2 style="margin: 10px 0; color: #999;">0 matches</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Display selected section details
    if 'show_section' in st.session_state:
        st.markdown("---")
        section = st.session_state['show_section']
        
        if section == 'bad_employer' and not bad_employer.empty:
            st.subheader("🏢 Bad Employer Details")
            display_donor_list(bad_employer)
        elif section == 'high_confidence' and not high_conf.empty:
            st.subheader("🔴 High Confidence Details")
            display_donor_list(high_conf)
        elif section == 'medium_confidence' and not medium_conf.empty:
            st.subheader("🟡 Medium Confidence Details")
            display_donor_list(medium_conf)

def display_donor_list(donors_df):
    """Display a list of donors with flag information in titles"""
    for _, donor in donors_df.iterrows():
        # Get clean flag description for title
        flag_desc = clean_flag_description(donor.get('flag_sources', ''))
        confidence = donor.get('confidence_level', '')
        flags = donor.get('flags', '')
        
        # Create title with flag information
        if 'BAD_EMPLOYER' in flags:
            icon = '🏢'
            title = f"{icon} {donor['first_name']} {donor['last_name']} ({donor['state']}) - {flag_desc}"
        else:
            icon = {'HIGH': '🔴', 'MEDIUM': '🟡'}.get(confidence, '🚨')
            title = f"{icon} {donor['first_name']} {donor['last_name']} ({donor['state']}) - {flag_desc}"
        
        if donor.get('is_grouped', False):
            title += f" [{donor.get('contribution_count', 1)} contributions]"
        
        with st.expander(title):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Name:** {donor['first_name']} {donor['last_name']}")
                st.write(f"**State:** {donor['state']}")
                if 'zip_code' in donor and donor['zip_code']:
                    st.write(f"**Zip Code:** {donor['zip_code']}")
                st.write(f"**Total Amount:** {donor['amount']}")
                if donor.get('contribution_count', 1) > 1:
                    st.write(f"**Contributions:** {donor['contribution_count']}")
                
                # Show enhanced contribution details if available
                if donor.get('contribution_dates') and len(donor['contribution_dates']) > 1:
                    dates_str = ", ".join(donor['contribution_dates'])
                    st.write(f"**Contribution Dates:** {dates_str}")
                    
                    if donor.get('contribution_amounts'):
                        amounts_str = ", ".join([f"{amt:.0f}" for amt in donor['contribution_amounts']])
                        st.write(f"**Contribution Amounts:** {amounts_str}")
                else:
                    st.write(f"**Date(s):** {donor['date']}")
            with col2:
                st.write(f"**Employer:** {donor['employer']}")
                if confidence:
                    st.write(f"**Confidence:** {confidence}")
                
                # Show RGA donation total if available
                if donor.get('rga_total') and donor['rga_total'] > 0:
                    st.write(f"**RGA Donations Total:** ${donor['rga_total']:,.2f}")
                    if donor.get('rga_contributions', 0) > 1:
                        st.write(f"**RGA Contributions:** {donor['rga_contributions']}")
                
                if donor.get('memo') and donor['memo']:
                    st.write(f"**Memo:** {donor['memo']}")
                    
                # Add investigation notes
                if 'BAD_EMPLOYER' in flags:
                    st.error("🚨 **URGENT** - Systemic corruption indicator")
                elif confidence == 'HIGH':
                    st.success("✅ **High confidence** - Exact match")
                elif confidence == 'MEDIUM':
                    st.warning("⚠️ **Medium confidence** - Verify location")

def display_flagged_donors_by_priority(flagged_donors):
    """Display flagged donors organized by priority sections"""
    if flagged_donors.empty:
        return
    
    # Group by priority (exclude low confidence)
    bad_employer = flagged_donors[flagged_donors['flags'].str.contains('BAD_EMPLOYER', na=False)]
    high_conf = flagged_donors[flagged_donors['confidence_level'] == 'HIGH']
    medium_conf = flagged_donors[flagged_donors['confidence_level'] == 'MEDIUM']
    
    # Display each section (exclude low confidence)
    sections = [
        ("🏢 Bad Employer Matches", bad_employer, "These indicate systemic corruption - highest priority"),
        ("🔴 High Confidence Matches", high_conf, "Exact name and state matches - very reliable"),
        ("🟡 Medium Confidence Matches", medium_conf, "Name matches - verify location")
    ]
    
    for section_title, section_data, section_help in sections:
        if not section_data.empty:
            st.subheader(section_title)
            st.caption(section_help)
            
            for _, donor in section_data.iterrows():
                display_individual_donor(donor)

def display_individual_donor(donor):
    """Display individual donor with all details"""
    # Create title with confidence level and grouping info
    confidence = donor.get('confidence_level', '')
    flags = donor.get('flags', '')
    
    if 'BAD_EMPLOYER' in flags:
        confidence_icon = '🏢'
        title_suffix = "BAD EMPLOYER"
    else:
        confidence_icon = {'HIGH': '🔴', 'MEDIUM': '🟡'}.get(confidence, '🚨')
        title_suffix = f"{confidence} CONFIDENCE" if confidence else "FLAGGED"
    
    title = f"{confidence_icon} {donor['first_name']} {donor['last_name']} ({donor['state']}) - {title_suffix}"
    if donor.get('is_grouped', False):
        title += f" [{donor.get('contribution_count', 1)} contributions]"
    
    with st.expander(title):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {donor['first_name']} {donor['last_name']}")
            st.write(f"**State:** {donor['state']}")
            if 'zip_code' in donor and donor['zip_code']:
                st.write(f"**Zip Code:** {donor['zip_code']}")
            st.write(f"**Total Amount:** {donor['amount']}")
            if donor.get('contribution_count', 1) > 1:
                st.write(f"**Number of Contributions:** {donor['contribution_count']}")
            st.write(f"**Date(s):** {donor['date']}")
        with col2:
            st.write(f"**Employer:** {donor['employer']}")
            if confidence:
                st.write(f"**Match Confidence:** {confidence}")
            st.write(f"**Flags:** {donor['flags']}")
            
            # Display sources with better formatting
            if donor['flag_sources']:
                st.write("**Flag Details:**")
                for source in donor['flag_sources'].split('|'):
                    st.write(f"• {source}")
            
            if donor['memo']:
                st.write(f"**Memo:** {donor['memo']}")
                
            # Add confidence explanation
            if 'BAD_EMPLOYER' in flags:
                st.error("🚨 **URGENT** - Bad employer match indicates systemic corruption")
            elif confidence == 'HIGH':
                st.success("✅ **High confidence match** - Exact name and state match")
            elif confidence == 'MEDIUM':
                st.warning("⚠️ **Medium confidence** - Name match, verify location")

def display_pac_dashboard_cards(pac_general_df):
    """Display PAC dashboard summary cards for different flag types"""
    if pac_general_df.empty:
        st.info("No flagged PACs found.")
        return
    
    # Group PACs by flag type
    bad_groups = pac_general_df[pac_general_df['flags'].str.contains('BAD_GROUP', na=False)]
    industry = pac_general_df[pac_general_df['flags'].str.contains('INDUSTRY', na=False)]
    lpac = pac_general_df[pac_general_df['flags'].str.contains('LPAC', na=False)]
    
    # Calculate totals
    def calc_pac_totals(df):
        if df.empty:
            return 0, 0.0
        count = len(df)
        # Convert amount to numeric for totaling
        amounts = pd.to_numeric(
            df['amount'].str.replace('$', '').str.replace(',', ''), 
            errors='coerce'
        ).fillna(0)
        total = amounts.sum()
        return count, total
    
    bad_groups_count, bad_groups_total = calc_pac_totals(bad_groups)
    industry_count, industry_total = calc_pac_totals(industry)
    lpac_count, lpac_total = calc_pac_totals(lpac)
    
    # Create PAC dashboard cards
    st.subheader("🏢 PAC Investigation Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if bad_groups_count > 0:
            st.markdown(f"""
            <div style="border: 2px solid #ff4b4b; border-radius: 10px; padding: 20px; text-align: center; background-color: #fff5f5;">
                <h3 style="color: #ff4b4b; margin: 0;">🚫 Bad Groups</h3>
                <h2 style="margin: 10px 0;">{bad_groups_count} matches</h2>
                <p style="margin: 5px 0; font-size: 18px;">${bad_groups_total:,.2f} total</p>
                <p style="margin: 0; color: #666; font-size: 14px;">Problematic organizations</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View Bad Groups Details", key="bad_groups", use_container_width=True):
                st.session_state['show_pac_section'] = 'bad_groups'
        else:
            st.markdown("""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center; background-color: #f9f9f9;">
                <h3 style="color: #999; margin: 0;">🚫 Bad Groups</h3>
                <h2 style="margin: 10px 0; color: #999;">0 matches</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if industry_count > 0:
            st.markdown(f"""
            <div style="border: 2px solid #ffa726; border-radius: 10px; padding: 20px; text-align: center; background-color: #fff8f0;">
                <h3 style="color: #ffa726; margin: 0;">🏭 Industry</h3>
                <h2 style="margin: 10px 0;">{industry_count} matches</h2>
                <p style="margin: 5px 0; font-size: 18px;">${industry_total:,.2f} total</p>
                <p style="margin: 0; color: #666; font-size: 14px;">Corporate classifications</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View Industry Details", key="industry", use_container_width=True):
                st.session_state['show_pac_section'] = 'industry'
        else:
            st.markdown("""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center; background-color: #f9f9f9;">
                <h3 style="color: #999; margin: 0;">🏭 Industry</h3>
                <h2 style="margin: 10px 0; color: #999;">0 matches</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if lpac_count > 0:
            st.markdown(f"""
            <div style="border: 2px solid #66bb6a; border-radius: 10px; padding: 20px; text-align: center; background-color: #f5fff5;">
                <h3 style="color: #66bb6a; margin: 0;">🏛️ LPAC</h3>
                <h2 style="margin: 10px 0;">{lpac_count} matches</h2>
                <p style="margin: 5px 0; font-size: 18px;">${lpac_total:,.2f} total</p>
                <p style="margin: 0; color: #666; font-size: 14px;">Leadership PACs</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View LPAC Details", key="lpac", use_container_width=True):
                st.session_state['show_pac_section'] = 'lpac'
        else:
            st.markdown("""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center; background-color: #f9f9f9;">
                <h3 style="color: #999; margin: 0;">🏛️ LPAC</h3>
                <h2 style="margin: 10px 0; color: #999;">0 matches</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Display selected section details
    if 'show_pac_section' in st.session_state:
        st.markdown("---")
        section = st.session_state['show_pac_section']
        
        if section == 'bad_groups' and not bad_groups.empty:
            st.subheader("🚫 Bad Groups Details")
            display_bad_groups_detail(bad_groups)
        elif section == 'industry' and not industry.empty:
            st.subheader("🏭 Industry Details")
            display_industry_detail(industry)
        elif section == 'lpac' and not lpac.empty:
            st.subheader("🏛️ LPAC Details")
            display_lpac_detail(lpac)

def group_pac_data_by_name(pac_df):
    """Group PAC data by name, combining contributions like donor grouping"""
    if pac_df.empty:
        return pac_df
    
    # Create grouping key based on PAC name
    pac_df['grouping_key'] = pac_df['pac_name'].str.upper().str.strip()
    
    # Group by the key and aggregate data
    grouped_data = []
    
    for group_key, group_df in pac_df.groupby('grouping_key'):
        if group_df.empty:
            continue
            
        # Use data from first record as base
        first_record = group_df.iloc[0].copy()
        
        # Aggregate amounts
        amounts = pd.to_numeric(
            group_df['amount'].str.replace('$', '').str.replace(',', ''), 
            errors='coerce'
        ).fillna(0)
        total_amount = amounts.sum()
        contribution_count = len(group_df)
        
        # Combine dates (show date range)
        dates = group_df['date'].dropna().tolist()
        if len(dates) > 1:
            date_range = f"{min(dates)} to {max(dates)}"
        elif len(dates) == 1:
            date_range = dates[0]
        else:
            date_range = ""
        
        # Combine memos (unique values only)
        memos = group_df['memo'].dropna().unique().tolist()
        combined_memo = " | ".join([memo for memo in memos if memo.strip()])
        
        # Create grouped record
        grouped_record = first_record.to_dict()
        grouped_record.update({
            'amount': f"${total_amount:,.2f}",
            'amount_numeric': total_amount,
            'date': date_range,
            'memo': combined_memo[:500] if combined_memo else first_record['memo'],
            'contribution_count': contribution_count,
            'is_grouped': contribution_count > 1
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

def format_org_type_display(org_code):
    """Convert single letter organization type to full name"""
    org_mapping = {
        'C': 'Corporation',
        'W': 'Corporation without capital stock',
        'L': 'Labor organization',
        'M': 'Membership organization',
        'T': 'Trade association',
        'V': 'Cooperative'
    }
    return org_mapping.get(str(org_code).upper(), f"No Org Type ({org_code})" if org_code else "No Org Type")

def display_bad_groups_detail(bad_groups_df):
    """Display Bad Groups with grouped PAC data and flag information"""
    # Group by PAC name
    grouped_pacs = group_pac_data_by_name(bad_groups_df)
    
    for _, pac in grouped_pacs.iterrows():
        # Extract flag from sources
        flag_desc = "Unknown Flag"
        if pac['flag_sources']:
            sources = pac['flag_sources'].split('|')
            for source in sources:
                if '🚫 Bad Group' in source:
                    flag_desc = source.split(': ')[-1] if ': ' in source else source
                    break
        
        # Create expandable with flag
        title = f"🚫 {pac['pac_name']} - {flag_desc} - {pac['amount']}"
        if pac.get('is_grouped', False):
            title += f" [{pac.get('contribution_count', 1)} contributions]"
        
        with st.expander(title):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**PAC Name:** {pac['pac_name']}")
                st.write(f"**Flag:** {flag_desc}")
                st.write(f"**Total Amount:** {pac['amount']}")
                if pac.get('contribution_count', 1) > 1:
                    st.write(f"**Contributions:** {pac['contribution_count']}")
            with col2:
                st.write(f"**Committee ID:** {pac['committee_id']}")
                st.write(f"**Date(s):** {pac['date']}")
                if pac['memo']:
                    st.write(f"**Memo:** {pac['memo']}")

def display_industry_detail(industry_df):
    """Display Industry data with summary tables and individual dropdowns"""
    from data_processor import FECDataProcessor
    processor = FECDataProcessor()
    
    # Get enhanced industry data
    enhanced_industry_data = []
    for _, row in industry_df.iterrows():
        enhanced_data = row.to_dict()
        
        # Get full industry data including org_type
        if row['committee_id']:
            full_data = processor.get_industry_full_data_by_id(row['committee_id'])
            if full_data:
                enhanced_data.update(full_data)
        
        # If no ID data, try fuzzy name matching
        if 'org_type' not in enhanced_data and row['pac_name']:
            full_data = processor.get_industry_full_data_by_name_fuzzy(row['pac_name'])
            if full_data:
                enhanced_data.update(full_data)
        
        enhanced_industry_data.append(enhanced_data)
    
    enhanced_df = pd.DataFrame(enhanced_industry_data)
    
    # 1. Industry Summary Table
    st.subheader("📊 Industry Category Totals")
    if 'larger_categories' in enhanced_df.columns:
        industry_summary = []
        for category in enhanced_df['larger_categories'].dropna().unique():
            cat_data = enhanced_df[enhanced_df['larger_categories'] == category]
            amounts = pd.to_numeric(
                cat_data['amount'].str.replace('$', '').str.replace(',', ''), 
                errors='coerce'
            ).fillna(0)
            industry_summary.append({
                'Industry Category': category,
                'Count': len(cat_data),
                'Total Amount': f"${amounts.sum():,.2f}"
            })
        
        if industry_summary:
            st.dataframe(pd.DataFrame(industry_summary), use_container_width=True)
    
    # 2. Organization Type Summary Table
    st.subheader("🏢 Organization Type Totals")
    if 'org_type' in enhanced_df.columns:
        org_summary = []
        for org_type in enhanced_df['org_type'].dropna().unique():
            org_data = enhanced_df[enhanced_df['org_type'] == org_type]
            amounts = pd.to_numeric(
                org_data['amount'].str.replace('$', '').str.replace(',', ''), 
                errors='coerce'
            ).fillna(0)
            org_summary.append({
                'Organization Type': format_org_type_display(org_type),
                'Count': len(org_data),
                'Total Amount': f"${amounts.sum():,.2f}"
            })
        
        if org_summary:
            st.dataframe(pd.DataFrame(org_summary), use_container_width=True)
    
    # 3. Individual PAC Dropdowns
    st.subheader("🔍 Individual PAC Details")
    grouped_pacs = group_pac_data_by_name(enhanced_df)
    
    for _, pac in grouped_pacs.iterrows():
        org_type_display = format_org_type_display(pac.get('org_type', ''))
        industry_display = pac.get('larger_categories', 'No Industry')
        
        title = f"🏭 {pac['pac_name']} - {pac['amount']} - {org_type_display}"
        if pac.get('is_grouped', False):
            title += f" [{pac.get('contribution_count', 1)} contributions]"
        
        with st.expander(title):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**PAC Name:** {pac['pac_name']}")
                st.write(f"**Industry:** {industry_display}")
                st.write(f"**Org Type:** {org_type_display}")
                st.write(f"**Total Amount:** {pac['amount']}")
            with col2:
                st.write(f"**Committee ID:** {pac.get('committee_id', 'No Committee ID')}")
                st.write(f"**Date(s):** {pac['date']}")
                if pac.get('contribution_count', 1) > 1:
                    st.write(f"**Contributions:** {pac['contribution_count']}")
                if pac.get('memo'):
                    st.write(f"**Memo:** {pac['memo']}")

def display_lpac_detail(lpac_df):
    """Display LPAC data with sponsor district information"""
    from data_processor import FECDataProcessor
    processor = FECDataProcessor()
    
    # Group by PAC name
    grouped_pacs = group_pac_data_by_name(lpac_df)
    
    for _, pac in grouped_pacs.iterrows():
        # Get sponsor district if available
        sponsor_district = "No Sponsor District"
        if pac['committee_id']:
            district = processor.get_lpac_full_data_by_id(pac['committee_id'])
            if district:
                sponsor_district = district
        
        title = f"🏛️ {pac['pac_name']} - {pac['amount']} - {sponsor_district}"
        if pac.get('is_grouped', False):
            title += f" [{pac.get('contribution_count', 1)} contributions]"
        
        with st.expander(title):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**PAC Name:** {pac['pac_name']}")
                st.write(f"**Sponsor District:** {sponsor_district}")
                st.write(f"**Total Amount:** {pac['amount']}")
                if pac.get('contribution_count', 1) > 1:
                    st.write(f"**Contributions:** {pac['contribution_count']}")
            with col2:
                st.write(f"**Committee ID:** {pac['committee_id']}")
                st.write(f"**Date(s):** {pac['date']}")
                if pac['memo']:
                    st.write(f"**Memo:** {pac['memo']}")

def create_us_state_choropleth(state_totals_df):
    """Create US state choropleth map showing contribution amounts"""
    if state_totals_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", 
                          xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=16))
        fig.update_layout(title="US State Contributions", title_x=0.5)
        return fig
    
    # Ensure state codes are properly formatted (2-letter uppercase)
    df_clean = state_totals_df.copy()
    df_clean['state'] = df_clean['state'].str.upper().str.strip()
    
    # Filter out invalid state codes (should be exactly 2 characters)
    df_clean = df_clean[df_clean['state'].str.len() == 2]
    
    if df_clean.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid state data available", 
                          xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=16))
        fig.update_layout(title="US State Contributions", title_x=0.5)
        return fig
    
    # Create hover text
    df_clean['hover_text'] = (
        df_clean['state'] + '<br>' +
        'Contributors: ' + df_clean['contributor_count'].astype(str) + '<br>' +
        'Amount: $' + df_clean['total_amount'].apply(lambda x: f"{x:,.2f}") + '<br>' +
        'Percentage: ' + df_clean['percentage'].apply(lambda x: f"{x:.1f}%")
    )
    
    # Create choropleth map
    fig = px.choropleth(
        df_clean,
        locations='state',
        color='total_amount',
        locationmode='USA-states',
        scope='usa',
        title='Campaign Contributions by State',
        color_continuous_scale='Reds',
        hover_name='state',
        hover_data={
            'state': False,
            'total_amount': ':$,.2f',
            'contributor_count': ':,',
            'percentage': ':.1f'
        },
        labels={
            'total_amount': 'Total Amount',
            'contributor_count': 'Contributors',
            'percentage': 'Percentage (%)'
        }
    )
    
    fig.update_layout(
        title_x=0.5,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='darkblue',
            coastlinewidth=1,
            showland=True,
            landcolor='rgba(248, 248, 248, 0.8)',
            showocean=True,
            oceancolor='rgba(173, 216, 230, 0.6)',
            showlakes=True,
            lakecolor='rgba(173, 216, 230, 0.8)',
            showrivers=True,
            rivercolor='rgba(173, 216, 230, 0.9)',
            riverwidth=1,
            showcountries=True,
            countrycolor='darkgray',
            countrywidth=1,
            projection_type='albers usa',
            bgcolor='rgba(255, 255, 255, 0)'
        ),
        coloraxis_colorbar=dict(
            title="Total Amount ($)",
            tickformat="$,.0f"
        ),
        height=600,  # Increased height for better detail
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_zip_choropleth(zip_data_df, state_code='CA', max_zips=500):
    """Create ZIP code choropleth heatmap using ZCTA boundary data"""
    if zip_data_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No ZIP code data available", 
                          xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=16))
        fig.update_layout(title=f"ZIP Code Contributions - {state_code}", title_x=0.5)
        return fig
    
    # Clean and prepare data
    df_clean = zip_data_df.copy()
    df_clean['zip_code'] = df_clean['zip_code'].astype(str).str.strip().str.zfill(5)
    
    # Filter out invalid ZIP codes
    df_clean = df_clean[df_clean['zip_code'].str.match(r'^\d{5}$')]
    
    # Limit data for performance
    if len(df_clean) > max_zips:
        df_clean = df_clean.nlargest(max_zips, 'total_amount')
    
    if df_clean.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid ZIP code data available", 
                          xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=16))
        fig.update_layout(title=f"ZIP Code Contributions - {state_code}", title_x=0.5)
        return fig
    
    # State-specific file mapping and map centers
    state_files = {
        'CA': ('ca_california_zip_codes_geo.min.json', 37.0, -119.4, 5.5),
        # Add more states as GeoJSON files become available
    }
    
    if state_code.upper() not in state_files:
        st.warning(f"ZIP boundary data not available for {state_code}. Showing scatter plot instead.")
        return create_state_zip_map(zip_data_df, state_code)
    
    filename, center_lat, center_lon, zoom_level = state_files[state_code.upper()]
    geojson_path = os.path.join("Databases", filename)
    
    try:
        import json
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        # Fallback to scatter plot if GeoJSON not available
        st.warning(f"ZIP boundary data not available for {state_code}. Showing scatter plot instead.")
        return create_state_zip_map(zip_data_df, state_code)
    
    # Create hover text
    df_clean['hover_text'] = (
        'ZIP: ' + df_clean['zip_code'] + '<br>' +
        'Contributors: ' + df_clean['contributor_count'].astype(str) + '<br>' +
        'Amount: $' + df_clean['total_amount'].apply(lambda x: f"{x:,.2f}")
    )
    
    # Create choropleth map using mapbox
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson_data,
        locations=df_clean['zip_code'],
        z=df_clean['total_amount'],
        featureidkey="properties.ZCTA5CE10",  # This is the ZIP code field in the GeoJSON
        colorscale='Reds',
        marker_opacity=0.7,
        marker_line_width=0.5,
        marker_line_color='white',
        hovertemplate='<b>%{text}</b><extra></extra>',
        text=df_clean['hover_text'],
        colorbar=dict(
            title="Total Amount ($)",
            tickformat="$,.0f"
        )
    ))
    
    fig.update_layout(
        title=f'Campaign Contributions by ZIP Code - {state_code}',
        title_x=0.5,
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_level
        ),
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def get_zip_coordinates(zip_code):
    """Get latitude and longitude for a zip code using comprehensive database"""
    # Handle ZIP+4 format by truncating to 5 digits
    if not zip_code or not str(zip_code).strip():
        return None
    
    # Clean and truncate zip code to 5 digits
    zip_clean = str(zip_code).strip()[:5]
    
    # Skip if not a valid 5-digit number
    if not zip_clean.isdigit() or len(zip_clean) != 5:
        return None
    
    # Load comprehensive ZIP coordinate database
    zip_coords = load_zip_coordinates()
    
    # Return coordinates if found, None otherwise
    return zip_coords.get(zip_clean, None)

def create_state_zip_map(zip_data_df, filer_state):
    """Create geographic map of zip codes within filer state"""
    if zip_data_df.empty or not filer_state:
        fig = go.Figure()
        fig.add_annotation(text="No state data available", 
                          xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=16))
        fig.update_layout(title=f"{filer_state} Geographic Distribution", title_x=0.5)
        return fig
    
    # Filter for filer state only
    state_zip_data = zip_data_df[zip_data_df['state'].str.upper() == filer_state.upper()].copy()
    
    if state_zip_data.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No zip code data available for {filer_state}", 
                          xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=16))
        fig.update_layout(title=f"{filer_state} Geographic Distribution", title_x=0.5)
        return fig
    
    # Get coordinates for zip codes
    coords_data = []
    for _, row in state_zip_data.iterrows():
        coords = get_zip_coordinates(row['zip_code'])
        if coords:
            coords_data.append({
                'zip_code': row['zip_code'],
                'lat': coords[0],
                'lon': coords[1],
                'total_amount': row['total_amount'],
                'contributor_count': row['contributor_count']
            })
    
    # Calculate coverage percentage
    coverage_percent = len(coords_data) / len(state_zip_data) * 100 if len(state_zip_data) > 0 else 0
    
    # Use fallback if coverage is too low (less than 10%)
    if not coords_data or coverage_percent < 10:
        # Enhanced fallback bar chart with better organization
        fallback_data = state_zip_data.head(20).copy()  # Top 20 zip codes
        
        # Truncate zip codes for display
        fallback_data['zip_display'] = fallback_data['zip_code'].astype(str).str[:5]
        
        fig = px.bar(
            fallback_data,
            x='zip_display',
            y='total_amount',
            title=f'Top Contributing Areas in {filer_state} (Geographic data: {coverage_percent:.0f}% coverage)',
            hover_data={
                'contributor_count': True,
                'zip_code': True,
                'zip_display': False
            },
            labels={
                'zip_display': 'Zip Code',
                'total_amount': 'Total Contributions ($)'
            }
        )
        
        fig.update_layout(
            title_x=0.5,
            xaxis_title="Zip Code (5-digit)",
            yaxis_title="Total Contributions ($)",
            xaxis_tickangle=-45,
            height=500
        )
        
        # Add note about geographic coverage
        fig.add_annotation(
            text=f"Geographic map unavailable - showing top areas by contribution amount",
            xref="paper", yref="paper",
            x=0.5, y=0.95, showarrow=False,
            font=dict(size=10, color="gray")
        )
        
        return fig
    
    # Create DataFrame for mapping
    coords_df = pd.DataFrame(coords_data)
    
    # Create enhanced scatter map with terrain styling
    fig = px.scatter_geo(
        coords_df,
        lat='lat',
        lon='lon',
        size='total_amount',
        color='total_amount',
        color_continuous_scale='Viridis',
        size_max=30,
        hover_name='zip_code',
        hover_data={
            'lat': False,
            'lon': False,
            'total_amount': ':$,.2f',
            'contributor_count': ':,'
        },
        title=f'{filer_state} Geographic Distribution of Contributions',
        labels={
            'total_amount': 'Total Amount',
            'contributor_count': 'Contributors'
        }
    )
    
    # Enhanced geo styling with useful detail features
    geo_config = dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='darkblue',
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue',
        showsubunits=True,        # State boundaries for geographic context
        subunitcolor='gray',      # Gray state boundary lines
        subunitwidth=1,          # Thin state boundary lines
        showlakes=True,          # Major lakes for geographic reference
        lakecolor='lightblue',   # Lake color matching ocean
        showrivers=True,         # Major rivers for geographic reference
        rivercolor='blue',       # Blue river lines
        riverwidth=0.5,         # Thin river lines
        resolution=50,           # Higher resolution for state-level detail
        scope='usa',
        projection_type='albers usa'
    )
    
    # Use standard USA scope for all states - no complex coordinate ranges
    
    fig.update_geos(**geo_config)
    
    fig.update_layout(
        title_x=0.5,
        height=600,  # Increased height for better detail
        coloraxis_colorbar=dict(
            title="Contribution Amount ($)"
        )
    )
    
    # Add major city labels for geographic reference
    major_cities = load_major_cities(state_filter=filer_state, population_threshold=30000, max_cities=20)
    if major_cities:
        city_lats = [city['lat'] for city in major_cities]
        city_lons = [city['lon'] for city in major_cities]
        city_names = [city['city'] for city in major_cities]
        
        fig.add_trace(go.Scattergeo(
            lon=city_lons,
            lat=city_lats,
            text=city_names,
            mode='text',
            textfont=dict(size=10, color='black', family='Arial'),
            textposition='middle center',
            showlegend=False,
            hoverinfo='skip'  # Don't show hover for city labels
        ))
    
    # Add coverage information
    fig.add_annotation(
        text=f"Showing {len(coords_data)} of {len(state_zip_data)} zip codes ({coverage_percent:.0f}% geographic coverage)",
        xref="paper", yref="paper",
        x=0.02, y=0.02, showarrow=False,
        font=dict(size=10, color="gray")
    )
    
    return fig

def create_folium_state_map(zip_data_df, filer_state):
    """Create interactive Folium map with built-in city labels and roads"""
    
    # Default center coordinates for all US states, DC, and territories
    state_centers = {
        'AL': [32.3617, -86.2792], 'AK': [64.0685, -152.2782], 'AZ': [34.2744, -111.2847],
        'AR': [34.7519, -92.1312], 'CA': [36.7783, -119.4179], 'CO': [39.5501, -105.7821],
        'CT': [41.6032, -73.0877], 'DE': [38.9108, -75.5277], 'FL': [27.7663, -82.4764],
        'GA': [32.1656, -82.9001], 'HI': [19.8968, -155.5828], 'ID': [44.0682, -114.7420],
        'IL': [40.6331, -89.3985], 'IN': [40.2732, -86.1349], 'IA': [42.0046, -93.2140],
        'KS': [38.4937, -98.3804], 'KY': [37.8393, -84.2700], 'LA': [30.3923, -92.4426],
        'ME': [45.2538, -69.4455], 'MD': [39.0458, -76.6413], 'MA': [42.2373, -71.5314],
        'MI': [44.3148, -85.6024], 'MN': [46.3287, -94.3053], 'MS': [32.3547, -89.3985],
        'MO': [37.9643, -91.8318], 'MT': [47.0527, -110.2148], 'NE': [41.4925, -99.9018],
        'NV': [38.4199, -117.1219], 'NH': [43.1939, -71.5724], 'NJ': [40.0583, -74.4057],
        'NM': [34.8405, -106.2485], 'NY': [42.1657, -74.9481], 'NC': [35.7596, -79.0193],
        'ND': [47.5515, -101.0020], 'OH': [40.4173, -82.9071], 'OK': [35.5889, -97.5348],
        'OR': [44.9319, -120.5542], 'PA': [40.2732, -76.8750], 'RI': [41.6762, -71.5562],
        'SC': [33.8191, -80.9066], 'SD': [44.2853, -100.2263], 'TN': [35.7449, -86.7489],
        'TX': [31.9686, -99.9018], 'UT': [39.3210, -111.0937], 'VT': [44.0582, -72.5806],
        'VA': [37.7693, -78.2057], 'WA': [47.3826, -121.0176], 'WV': [38.4680, -80.9696],
        'WI': [44.2853, -89.6385], 'WY': [42.7475, -107.2085], 'DC': [38.8974, -77.0365],
        # US Territories
        'PR': [18.2208, -66.5901], 'VI': [18.0001, -64.8199], 'GU': [13.4443, 144.7937],
        'AS': [-14.2710, -170.1322], 'MP': [17.3308, 145.3846]
    }
    
    # Get state center coordinates
    if filer_state and filer_state.upper() in state_centers:
        center_coords = state_centers[filer_state.upper()]
    else:
        center_coords = [39.8283, -98.5795]  # Geographic center of US
    
    # Create base map with built-in city labels and roads
    folium_map = folium.Map(
        location=center_coords,
        zoom_start=7,
        tiles='OpenStreetMap',  # Built-in city labels and road networks
    )
    
    # Add contribution data points if available
    if not zip_data_df.empty:
        # Filter for filer state only if specified
        if filer_state:
            state_zip_data = zip_data_df[zip_data_df['state'].str.upper() == filer_state.upper()].copy()
        else:
            state_zip_data = zip_data_df.copy()
        
        # Sort by contribution amount for better geographic distribution (highest first)
        state_zip_data = state_zip_data.sort_values('total_amount', ascending=False)
        
        # Calculate color scale parameters
        min_amount = state_zip_data['total_amount'].min()
        max_amount = state_zip_data['total_amount'].max()
        
        # Create Viridis-style LinearColormap for markers (no legend)
        colormap = cm.LinearColormap(
            colors=['#440154', '#3b518b', '#21908c', '#f97306', '#e31a1c'],
            vmin=min_amount,
            vmax=max_amount
        )
        
        # Add markers for contribution locations
        coords_added = 0
        for _, row in state_zip_data.iterrows():
            coords = get_zip_coordinates(row['zip_code'])
            if coords and coords_added < 2000:  # Increased limit for comprehensive coverage
                lat, lon = coords
                
                # Create popup with contribution info
                popup_text = f"""
                <b>ZIP: {row['zip_code']}</b><br>
                Contributors: {row['contributor_count']}<br>
                Total: ${row['total_amount']:,.2f}
                """
                
                # Smaller marker size based on contribution amount (2-8px radius)
                normalized_size = max(2, min(8, (row['total_amount'] / max_amount) * 6 + 2))
                
                # Get Viridis color from LinearColormap
                marker_color = colormap(row['total_amount'])
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=normalized_size,
                    popup=folium.Popup(popup_text, max_width=200),
                    color=marker_color,
                    fill=True,
                    fillColor=marker_color,
                    fillOpacity=0.7,
                    weight=1
                ).add_to(folium_map)
                
                coords_added += 1
        
        # Add custom centered color legend using proper Branca template
        from branca.element import Template, MacroElement
        
        legend_template = f'''
        {{% macro html(this, kwargs) %}}
        <div style="position: fixed; 
             bottom: 38px; 
             left: 50%; 
             transform: translateX(-50%); 
             width: 320px; 
             height: 80px; 
             background-color: white; 
             border: 2px solid #333; 
             z-index: 9999; 
             font-size: 12px;
             font-family: Arial, sans-serif;
             padding: 10px;
             opacity: 0.95;
             border-radius: 8px;
             box-shadow: 0 2px 5px rgba(0,0,0,0.3);">
        <b style="color: #000;">Contribution Amount ($)</b><br>
        <div style="background: linear-gradient(to right, #440154, #3b518b, #21908c, #f97306, #e31a1c); 
             height: 15px; 
             margin: 8px 0;
             border: 1px solid #ccc;
             border-radius: 3px;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #000;">
             <span>${min_amount:,.0f}</span>
             <span>Low → High</span>
             <span>${max_amount:,.0f}</span>
        </div>
        </div>
        {{% endmacro %}}
        '''
        
        # Create the legend macro element
        legend_macro = MacroElement()
        legend_macro._template = Template(legend_template)
        
        # Add the legend to the map
        folium_map.get_root().add_child(legend_macro)
    
    return folium_map


def display_geographic_summary_tables(state_totals_df, state_breakdown):
    """Display summary tables for geographic analysis"""
    
    # Top 10 States Table
    st.subheader("📊 Top 10 Contributing States")
    if not state_totals_df.empty:
        top_10_states = state_totals_df.head(10).copy()
        top_10_states['total_amount'] = top_10_states['total_amount'].apply(lambda x: f"${x:,.2f}")
        top_10_states['percentage'] = top_10_states['percentage'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            top_10_states[['state', 'contributor_count', 'total_amount', 'percentage']],
            column_config={
                'state': 'State',
                'contributor_count': 'Contributors',
                'total_amount': 'Total Amount',
                'percentage': 'Percentage'
            },
            use_container_width=True
        )
    else:
        st.info("No state data available")
    
    # In-State vs Out-of-State Breakdown
    st.subheader("🏠 In-State vs Out-of-State Analysis")
    if state_breakdown:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "In-State Contributions", 
                f"${state_breakdown['in_state_total']:,.2f}",
                f"{state_breakdown['in_state_percentage']:.1f}% of total"
            )
            st.write(f"**Contributors:** {state_breakdown['in_state_count']:,}")
        
        with col2:
            st.metric(
                "Out-of-State Contributions", 
                f"${state_breakdown['out_of_state_total']:,.2f}",
                f"{state_breakdown['out_of_state_percentage']:.1f}% of total"
            )
            st.write(f"**Contributors:** {state_breakdown['out_of_state_count']:,}")
    else:
        st.info("No breakdown data available")

def load_file_data(uploaded_file, skiprows=0):
    """Load data from uploaded file with robust error handling"""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Try multiple reading strategies for FEC files
            # Try to handle variable-width CSV files properly using pure CSV reader
            import csv
            
            try:
                uploaded_file.seek(0)
                content = uploaded_file.read().decode('utf-8')
                
                # Parse CSV with proper handling of variable column counts
                rows = []
                lines = content.split('\n')
                lines_to_process = lines[skiprows:] if skiprows > 0 else lines
                
                for line in lines_to_process:
                    if line.strip():
                        reader = csv.reader([line])
                        try:
                            row = next(reader)
                            rows.append(row)
                        except:
                            continue
                
                if rows:
                    # Find max columns and pad all rows
                    max_cols = max(len(row) for row in rows)
                    
                    # Only use this method if we detect many columns (FEC format)
                    if max_cols > 20:
                        for row in rows:
                            while len(row) < max_cols:
                                row.append('')
                        
                        # Create DataFrame
                        df = pd.DataFrame(rows)
                        
                        # Generate Excel-like column names
                        def generate_excel_columns(n):
                            columns = []
                            for i in range(n):
                                if i < 26:
                                    columns.append(chr(65 + i))  # A-Z
                                else:
                                    first = chr(65 + (i - 26) // 26)
                                    second = chr(65 + (i - 26) % 26)
                                    columns.append(first + second)
                            return columns
                        
                        df.columns = generate_excel_columns(len(df.columns))
                        return df
                        
            except Exception as e:
                # Fallback to original strategies
                pass
            
            # Build strategies with user-specified skiprows
            base_strategies = [
                # Strategy 1: User-specified skiprows with python engine
                {'params': {'header': None, 'quoting': 1, 'engine': 'python'}, 'desc': 'python engine, no headers, quoted fields'},
                # Strategy 2: Default reading with proper quoting
                {'params': {'quoting': 1}, 'desc': 'default headers with quoted fields'},
                # Strategy 3: Different encoding
                {'params': {'encoding': 'latin-1', 'quoting': 1}, 'desc': 'latin-1 encoding, quoted fields'},
                # Strategy 4: Tab separated
                {'params': {'sep': '\t'}, 'desc': 'tab separated'},
            ]
            
            strategies = []
            for strategy in base_strategies:
                # Add user-specified skiprows if > 0
                if skiprows > 0:
                    strategy_with_skip = strategy.copy()
                    strategy_with_skip['params'] = strategy['params'].copy()
                    strategy_with_skip['params']['skiprows'] = skiprows
                    strategy_with_skip['desc'] = f"skip {skiprows} rows, " + strategy['desc']
                    strategies.append(strategy_with_skip)
                
                # Also try original strategy without skiprows as fallback
                strategies.append(strategy)
            
            for i, strategy in enumerate(strategies):
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, **strategy['params'])
                    
                    if len(df) > 0 and len(df.columns) > 0:
                        # Validate this looks like FEC data
                        is_fec_data = False
                        if strategy['params'].get('header') is None:
                            # Check if first column contains SA values (FEC form types)
                            first_col_values = df.iloc[:, 0].astype(str).str.upper()
                            sa_count = first_col_values.str.contains('SA', na=False).sum()
                            if sa_count > 0:
                                is_fec_data = True
                                # Technical success message hidden for clean UI
                        
                        # If no header was used, generate column names A, B, C...Z, AA, AB, etc.
                        if strategy['params'].get('header') is None:
                            def generate_excel_columns(n):
                                """Generate Excel-style column names: A, B, C...Z, AA, AB, etc."""
                                columns = []
                                for i in range(n):
                                    if i < 26:
                                        columns.append(chr(65 + i))  # A-Z
                                    else:
                                        # AA, AB, AC, etc.
                                        first = chr(65 + (i - 26) // 26)
                                        second = chr(65 + (i - 26) % 26)
                                        columns.append(first + second)
                                return columns
                            
                            df.columns = generate_excel_columns(len(df.columns))
                            # Technical column info hidden for clean UI
                        
                        # Validate FEC data format
                        if strategy['params'].get('header') is None and len(df.columns) > 10:
                            # Quick validation that this looks like FEC data
                            first_col_values = df.iloc[:, 0].astype(str).str.upper()
                            sa_count = first_col_values.str.contains('SA', na=False).sum()
                            if sa_count == 0:
                                continue  # Try next strategy if no SA records found
                        
                        return df
                        
                except Exception as e:
                    if i < len(strategies) - 1:
                        continue  # Try next strategy
                    else:
                        raise e  # Last strategy failed, raise error
                        
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, skiprows=skiprows)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file with all strategies: {str(e)}")
        st.info("Please check that your file is a valid CSV or Excel file with FEC data.")
        return None

def display_results(results):
    """Display processing results with tabs for different data types"""
    
    bad_donors_df = results['bad_donors']
    pac_data_df = results['pac_data']
    
    # STREAMLINED VERSION: Show only Individual Donors and Geographic Analysis
    tab1, tab2, tab3 = st.tabs(["🚨 Flagged Individuals", "🗺️ Geographic Analysis", "📊 Summary"])
    
    # ORIGINAL CODE (COMMENTED OUT FOR RESTORATION):
    # tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚨 Flagged Individuals", "🏢 Committee/PAC Data", "🗳️ Bad Legislation", "🗺️ Geographic Analysis", "📊 Summary"])
    
    with tab1:
        st.subheader("Individual Donor Analysis")
        if not bad_donors_df.empty:
            # Show flagged donors with dashboard cards
            flagged_donors = bad_donors_df[bad_donors_df['flags'] != '']
            if not flagged_donors.empty:
                # Sort flagged donors by priority
                flagged_donors = sort_donors_by_priority(flagged_donors)
                
                # Display dashboard cards
                display_dashboard_cards(flagged_donors)
                
                # Generate email report for high confidence matches
                email_report = generate_email_report(flagged_donors)
                if email_report:
                    st.subheader("📧 Email Report")
                    st.text_area(
                        "Copy and paste into email:",
                        value=email_report,
                        height=120,
                        help="Ready-to-send email summary"
                    )
            
            # Show all individual donors in table
            st.subheader("All Individual Donors")
            st.dataframe(bad_donors_df, use_container_width=True)
            
            # Download button for individual donors
            csv_individuals = bad_donors_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Individual Donors CSV",
                data=csv_individuals,
                file_name="individual_donors_analysis.csv",
                mime="text/csv"
            )
        else:
            st.info("No individual donors found in the data.")
    
    with tab2:
        st.subheader("Geographic Analysis")
        if not bad_donors_df.empty:
            # Get the original DataFrame before processing results to extract filer info
            if 'original_df' in st.session_state and 'current_processor' in st.session_state:
                original_df = st.session_state['original_df']
                current_processor = st.session_state['current_processor']
                
                # Extract filer information and financial data
                filer_info = current_processor.extract_filer_info(original_df)
                filer_state = filer_info.get('filer_state')
                filer_district = filer_info.get('filer_district')
                committee_name = filer_info.get('committee_name')
                financial_data = filer_info.get('financial_data', {})
                
                # Calculate derived metrics for FEC data
                derived_metrics = {}
                if hasattr(current_processor, 'calculate_derived_metrics'):
                    derived_metrics = current_processor.calculate_derived_metrics(financial_data, bad_donors_df)
                
                # Store financial data in session state for use across tabs
                st.session_state['fec_financial_data'] = financial_data
                st.session_state['fec_derived_metrics'] = derived_metrics
                st.session_state['fec_committee_name'] = committee_name
                
                # For generic data, use user-selected focus state if available
                raw_focus_state = st.session_state.get('geographic_focus_state', '').upper().strip()
                
                # Validate the focus state
                valid_states = {
                    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
                    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
                }
                
                # Only use focus state if it's valid
                geographic_focus_state = raw_focus_state if raw_focus_state in valid_states else None
                display_state = filer_state or geographic_focus_state
                
                # Get geographic analysis
                geographic_data = current_processor.get_geographic_analysis(bad_donors_df)
                state_breakdown = current_processor.get_state_breakdown(bad_donors_df, filer_state)
                
                # Display filer/focus information
                if filer_state and filer_district:
                    st.info(f"**Filer:** {filer_state}-{filer_district}")
                elif filer_state:
                    st.info(f"**Filer State:** {filer_state}")
                elif geographic_focus_state:
                    st.info(f"**Geographic Focus State:** {geographic_focus_state}")
                
                # Create two maps side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🗺️ US State Overview")
                    if not geographic_data['state_totals'].empty:
                        us_map = create_us_state_choropleth(geographic_data['state_totals'])
                        st.plotly_chart(us_map, use_container_width=True)
                    else:
                        st.info("No state data available for mapping")
                
                with col2:
                    st.subheader(f"🏛️ {display_state} Geographic Detail" if display_state else "State Geographic Detail")
                    if not geographic_data['zip_data'].empty and display_state:
                        # Add map type selector
                        map_type = st.radio(
                            "Map Type:",
                            ["Interactive Points", "ZIP Heatmap"],
                            horizontal=True,
                            help="Choose between point-based interactive map or ZIP code heatmap visualization"
                        )
                        
                        if map_type == "ZIP Heatmap":
                            # Create ZIP code choropleth heatmap
                            if display_state.upper() == 'CA':  # Only California supported for now
                                zip_choropleth = create_zip_choropleth(geographic_data['zip_data'], display_state.upper())
                                st.plotly_chart(zip_choropleth, use_container_width=True)
                                st.caption("🔥 ZIP code contribution heatmap using Census boundary data")
                            else:
                                st.warning(f"ZIP heatmap not yet available for {display_state}. Only California (CA) is currently supported.")
                                # Fallback to regular scatter plot
                                zip_map = create_state_zip_map(geographic_data['zip_data'], display_state)
                                st.plotly_chart(zip_map, use_container_width=True)
                        else:
                            # Create Folium map with built-in city labels and roads
                            folium_map = create_folium_state_map(geographic_data['zip_data'], display_state)
                            st_folium(folium_map, width=700, height=500)
                            
                            # Add info about the enhanced features
                            st.caption("🗺️ Interactive map with built-in city labels, roads, and street networks")
                        
                        # Add state dropdown for quick switching
                        available_states = get_available_states_for_dropdown(geographic_data)
                        if available_states:
                            st.markdown("---")
                            current_display = f"Current: {display_state}" if display_state else "No state selected"
                            
                            # Create options list with current state at top
                            dropdown_options = [current_display] + available_states
                            
                            def on_state_change():
                                """Callback for state dropdown change"""
                                selected = st.session_state.state_dropdown_switcher
                                if selected != current_display:
                                    new_state = selected[:2]
                                    st.session_state['geographic_focus_state'] = new_state
                            
                            selected_state_option = st.selectbox(
                                "🗺️ Quick Switch to Another State:",
                                options=dropdown_options,
                                key="state_dropdown_switcher",
                                help="Select a different state to view its contribution map",
                                on_change=on_state_change
                            )
                        
                    elif not geographic_data['zip_data'].empty:
                        st.info("💡 Tip: Enter a focus state above to see ZIP code detail map")
                    else:
                        st.info("No zip code data available for state mapping")
                
                # Display summary tables
                st.markdown("---")
                display_geographic_summary_tables(geographic_data['state_totals'], state_breakdown)
            else:
                st.warning("Original file data not available. Please rerun the analysis.")
        else:
            st.info("No individual donor data available for geographic analysis.")
    
    with tab3:
        st.subheader("Analysis Summary")
        
        # Get target state from existing geographic focus state (used for ZIP maps)
        target_state = st.session_state.get('geographic_focus_state', '').upper().strip()
        if not target_state and 'filer_state' in locals() and filer_state:
            target_state = filer_state
        
        # Get target donation amount from session state (set in geographic settings)
        target_donation_amount = st.session_state.get('target_donation_amount', 2800)
        
        # Calculate enhanced metrics for state/local reports
        if not bad_donors_df.empty:
            # Get geographic data for calculations
            if 'current_processor' in locals():
                geographic_data = current_processor.get_geographic_analysis(bad_donors_df)
                state_totals = geographic_data['state_totals']
            else:
                state_totals = pd.DataFrame()
            
            # Calculate recurring donors (3+ contributions)
            recurring_donors = 0
            if 'contribution_count' in bad_donors_df.columns:
                recurring_donors = len(bad_donors_df[bad_donors_df['contribution_count'] >= 3])
            
            # Calculate median donation amount
            median_amount = 0
            if 'amount_numeric' in bad_donors_df.columns:
                median_amount = bad_donors_df['amount_numeric'].median()
            
            # Calculate max donors (within +/- $100 of target amount)
            max_donors_count = 0
            if 'amount_numeric' in bad_donors_df.columns:
                lower_bound = target_donation_amount - 100
                upper_bound = target_donation_amount + 100
                max_donors_count = len(bad_donors_df[
                    (bad_donors_df['amount_numeric'] >= lower_bound) & 
                    (bad_donors_df['amount_numeric'] <= upper_bound)
                ])
            
            # Calculate percentage of contributions in target state (by dollar amount)
            state_percentage = 0
            if target_state and not state_totals.empty:
                target_state_amount = state_totals[state_totals['state'] == target_state]['total_amount'].sum()
                total_amount = state_totals['total_amount'].sum()
                if total_amount > 0:
                    state_percentage = (target_state_amount / total_amount) * 100
            
            # Get top 3 donor states
            top_3_states = []
            if not state_totals.empty:
                top_3_states = state_totals.head(3)[['state', 'contributor_count']].values.tolist()
            
            # Count RGA donors - check both flags and flag_sources
            rga_donors_count = 0
            if 'flag_sources' in bad_donors_df.columns:
                # RGA donors are identified in flag_sources as 'RGA Donor'
                rga_donors_count = len(bad_donors_df[bad_donors_df['flag_sources'].str.contains('RGA Donor', na=False)])
            elif 'flags' in bad_donors_df.columns:
                # Fallback to checking flags column
                rga_donors_count = len(bad_donors_df[bad_donors_df['flags'].str.contains('RGA', na=False)])
        
        # Enhanced metrics row for state/local reports
        if 'file_format' in st.session_state and st.session_state.file_format == "State or Local Report":
            st.markdown("### 📈 Enhanced Donor Analytics")
            
            # First row of enhanced metrics: Total Donors, Median Donation, Max Donors, RGA Donors
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_individuals = len(bad_donors_df)
                total_contributions = bad_donors_df['contribution_count'].sum() if 'contribution_count' in bad_donors_df.columns else total_individuals
                st.metric(
                    "Total Donors", 
                    total_individuals,
                    help=f"Total contributions: {total_contributions}"
                )
            
            with col2:
                st.metric(
                    "Median Donation", 
                    f"${median_amount:,.2f}" if median_amount > 0 else "$0.00",
                    help="Median amount across all contributions"
                )
            
            with col3:
                lower_bound = target_donation_amount - 100
                upper_bound = target_donation_amount + 100
                st.metric(
                    "Max Donors", 
                    max_donors_count,
                    help=f"Donors who gave ${lower_bound:,.0f}-${upper_bound:,.0f}"
                )
            
            with col4:
                st.metric(
                    "RGA Donors", 
                    rga_donors_count,
                    help="Donors flagged in RGA database"
                )
            
            # Second row of enhanced metrics: Percent in state, Top 3 states
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                if target_state:
                    st.metric(
                        f"% Contributions in {target_state}", 
                        f"{state_percentage:.1f}%",
                        help=f"Percentage of total contribution dollars from {target_state}"
                    )
                else:
                    st.metric("% Contributions in State", "Set focus state", help="Set geographic focus state in column mapping section")
            
            with col6:
                if len(top_3_states) >= 1:
                    st.metric(
                        f"Top State: {top_3_states[0][0]}", 
                        f"{top_3_states[0][1]} donors",
                        help="State with most donors"
                    )
                else:
                    st.metric("Top State", "N/A")
            
            with col7:
                if len(top_3_states) >= 2:
                    st.metric(
                        f"2nd: {top_3_states[1][0]}", 
                        f"{top_3_states[1][1]} donors",
                        help="Second highest donor state"
                    )
                else:
                    st.metric("2nd State", "N/A")
            
            with col8:
                if len(top_3_states) >= 3:
                    st.metric(
                        f"3rd: {top_3_states[2][0]}", 
                        f"{top_3_states[2][1]} donors",
                        help="Third highest donor state"
                    )
                else:
                    st.metric("3rd State", "N/A")
            
            # Additional metrics row
            st.markdown("#### Additional Analytics")
            col9, col10, col11, col12 = st.columns(4)
            
            with col9:
                st.metric(
                    "Recurring Donors", 
                    recurring_donors,
                    help="Donors who gave 3 or more times"
                )
            
            # Donors over time line chart
            if 'date' in bad_donors_df.columns and not bad_donors_df['date'].isna().all():
                st.markdown("### 📈 Donors Over Time")
                
                # Convert dates and create time series
                df_with_dates = bad_donors_df.copy()
                df_with_dates['date'] = pd.to_datetime(df_with_dates['date'], errors='coerce')
                df_with_dates = df_with_dates.dropna(subset=['date'])
                
                if not df_with_dates.empty:
                    # Group by date and count unique donors
                    daily_donors = df_with_dates.groupby(df_with_dates['date'].dt.date).agg({
                        'first_name': 'count'  # Count contributions per day
                    }).reset_index()
                    daily_donors.columns = ['Date', 'Contributions']
                    
                    # Create cumulative donors over time
                    daily_donors = daily_donors.sort_values('Date')
                    daily_donors['Cumulative_Donors'] = daily_donors['Contributions'].cumsum()
                    
                    # Display both daily and cumulative charts
                    chart_type = st.radio("Chart Type:", ["Daily Contributions", "Cumulative Total"], horizontal=True)
                    
                    if chart_type == "Daily Contributions":
                        st.line_chart(daily_donors.set_index('Date')['Contributions'])
                        st.caption("📊 Number of contributions received each day")
                    else:
                        st.line_chart(daily_donors.set_index('Date')['Cumulative_Donors'])
                        st.caption("📈 Cumulative total of contributions over time")
                else:
                    st.info("No valid dates found for time series analysis")
            else:
                st.info("💡 Date information not available for time series analysis")
            
            st.markdown("---")
        
        # FEC Financial Summary Section (only for FEC data)
        if ('file_format' in st.session_state and 
            st.session_state.file_format == "FEC Quarterly Report" and
            'fec_financial_data' in st.session_state and
            'fec_derived_metrics' in st.session_state):
            
            financial_data = st.session_state['fec_financial_data']
            derived_metrics = st.session_state['fec_derived_metrics']
            
            st.markdown("### 💰 Financial Summary")
            
            # First row of financial metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Receipts", 
                    f"${financial_data.get('receipts', 0):,.2f}",
                    help="Total receipts from FEC filing"
                )
            
            with col2:
                st.metric(
                    "Disbursements", 
                    f"${financial_data.get('disbursements', 0):,.2f}",
                    help="Total disbursements from FEC filing"
                )
            
            with col3:
                st.metric(
                    "Cash on Hand", 
                    f"${financial_data.get('coh', 0):,.2f}",
                    help="Cash on hand at end of reporting period"
                )
            
            with col4:
                st.metric(
                    "Debt", 
                    f"${financial_data.get('debt', 0):,.2f}",
                    help="Outstanding debt"
                )
            
            # Second row of financial metrics
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric(
                    "Loans by Candidate", 
                    f"${financial_data.get('loans_by_candidate', 0):,.2f}",
                    help="Loans made by the candidate to the committee"
                )
            
            with col6:
                burn_rate = derived_metrics.get('burn_rate', 0)
                st.metric(
                    "Burn Rate", 
                    f"{burn_rate:.1f}%",
                    help="Disbursements as percentage of receipts"
                )
            
            with col7:
                small_donor_pct = derived_metrics.get('small_donor_percentage', 0)
                st.metric(
                    "% from Small Donors", 
                    f"{small_donor_pct:.1f}%",
                    help="Small donor contributions as percentage of total receipts"
                )
            
            with col8:
                max_out_donors = derived_metrics.get('max_out_donors', 0)
                st.metric(
                    "Max Out Donors", 
                    f"{max_out_donors:,}",
                    help="Donors who gave $3,400-$3,600 (max contribution range)"
                )
            
            # Third row - additional metrics
            col9, col10, col11, col12 = st.columns(4)
            
            with col9:
                total_contributions = derived_metrics.get('total_individual_contributions', 0)
                st.metric(
                    "Total Individual Contributions", 
                    f"{total_contributions:,}",
                    help="Total number of individual contributions"
                )
            
            with col10:
                median_contribution = derived_metrics.get('median_contribution', 0)
                st.metric(
                    "Median Contribution", 
                    f"${median_contribution:.2f}",
                    help="Median individual contribution amount"
                )
            
            # FEC Financial Email Report
            st.markdown("### 📧 Financial Email Report")
            fec_email_report = generate_fec_financial_email_report()
            if fec_email_report:
                st.text_area(
                    "Copy and paste financial summary:",
                    value=fec_email_report,
                    height=200,
                    help="Ready-to-send financial summary for email"
                )
            
            # Add time series for FEC reports too
            if 'date' in bad_donors_df.columns and not bad_donors_df['date'].isna().all():
                st.markdown("### 📈 Contributions Over Time")
                
                # Convert dates and create time series
                df_with_dates = bad_donors_df.copy()
                df_with_dates['date'] = pd.to_datetime(df_with_dates['date'], errors='coerce')
                df_with_dates = df_with_dates.dropna(subset=['date'])
                
                if not df_with_dates.empty:
                    # Group by date and count contributions
                    daily_contributions = df_with_dates.groupby(df_with_dates['date'].dt.date).agg({
                        'first_name': 'count'  # Count contributions per day
                    }).reset_index()
                    daily_contributions.columns = ['Date', 'Contributions']
                    
                    # Create cumulative contributions over time
                    daily_contributions = daily_contributions.sort_values('Date')
                    daily_contributions['Cumulative_Contributions'] = daily_contributions['Contributions'].cumsum()
                    
                    # Display both daily and cumulative charts
                    chart_type = st.radio("Chart Type:", ["Daily Contributions", "Cumulative Total"], horizontal=True, key="fec_chart_type")
                    
                    if chart_type == "Daily Contributions":
                        st.line_chart(daily_contributions.set_index('Date')['Contributions'])
                        st.caption("📊 Number of contributions received each day")
                    else:
                        st.line_chart(daily_contributions.set_index('Date')['Cumulative_Contributions'])
                        st.caption("📈 Cumulative total of contributions over time")
                else:
                    st.info("No valid dates found for time series analysis")
            else:
                st.info("💡 Date information not available for time series analysis")
            
            st.markdown("---")
        
        # Standard metrics row
        st.markdown("### 📊 Standard Analysis Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_individuals = len(bad_donors_df)
            total_contributions = bad_donors_df['contribution_count'].sum() if 'contribution_count' in bad_donors_df.columns else total_individuals
            st.metric("Unique Individual Donors", total_individuals, help=f"Total contributions: {total_contributions}")
        
        with col2:
            if not bad_donors_df.empty:
                flagged_individuals = len(bad_donors_df[bad_donors_df['flags'] != ''])
                high_conf = len(bad_donors_df[bad_donors_df['confidence_level'] == 'HIGH'])
                medium_conf = len(bad_donors_df[bad_donors_df['confidence_level'] == 'MEDIUM'])
                help_text = f"🔴 High: {high_conf} | 🟡 Medium: {medium_conf}"
                st.metric("Flagged Individuals", flagged_individuals, help=help_text)
            else:
                st.metric("Flagged Individuals", 0)
        
        with col3:
            total_committees = len(pac_data_df) if not pac_data_df.empty else 0
            st.metric("Total Committees/PACs", total_committees)
        
        with col4:
            flagged_committees = len(pac_data_df[pac_data_df['flags'] != '']) if not pac_data_df.empty else 0
            st.metric("Total Flagged Committees", flagged_committees)
        
        # Second row - PAC breakdown
        st.markdown("### Committee/PAC Breakdown")
        col5, col6, col7, col8 = st.columns(4)
        
        if not pac_data_df.empty:
            # Separate counts
            bad_legislation_count = len(pac_data_df[pac_data_df['flags'].str.contains('BAD_LEGISLATION', na=False)])
            general_pac_count = len(pac_data_df[~pac_data_df['flags'].str.contains('BAD_LEGISLATION', na=False) & (pac_data_df['flags'] != '')])
            bad_groups_count = len(pac_data_df[pac_data_df['flags'].str.contains('BAD_GROUP', na=False)])
            industry_count = len(pac_data_df[pac_data_df['flags'].str.contains('INDUSTRY', na=False)])
            lpac_count = len(pac_data_df[pac_data_df['flags'].str.contains('LPAC', na=False)])
        else:
            bad_legislation_count = general_pac_count = bad_groups_count = industry_count = lpac_count = 0
        
        with col5:
            st.metric("🗳️ Bad Legislation", bad_legislation_count, help="Politicians with concerning records")
        
        with col6:
            st.metric("🏢 General PACs", general_pac_count, help=f"Bad Groups: {bad_groups_count} | Industry: {industry_count} | LPAC: {lpac_count}")
        
        with col7:
            st.metric("🚫 Bad Groups", bad_groups_count, help="Problematic organizations")
        
        with col8:
            st.metric("🏭 Industry/LPAC", industry_count + lpac_count, help=f"Industry: {industry_count} | LPAC: {lpac_count}")
        
        # Flag type breakdown
        if not bad_donors_df.empty or not pac_data_df.empty:
            st.subheader("Flag Type Breakdown")
            
            all_flags = []
            if not bad_donors_df.empty:
                for flags in bad_donors_df['flags']:
                    if flags:
                        all_flags.extend(flags.split('|'))
            
            if not pac_data_df.empty:
                for flags in pac_data_df['flags']:
                    if flags:
                        all_flags.extend(flags.split('|'))
            
            if all_flags:
                flag_counts = pd.Series(all_flags).value_counts()
                st.bar_chart(flag_counts)

def main():
    st.title("🔍 FEC Donor & PAC Analysis Tool")
    st.markdown("**Analyze campaign finance data and flag suspicious patterns**")
    
    # Initialize database
    initialize_database()
    
    # Debug section (for troubleshooting RGA functionality)
    with st.sidebar.expander("🔧 Debug Tools", expanded=False):
        if st.button("Check CSV Files Status"):
            from config import CSV_FILES, BASE_DIR, DATABASE_PATH
            
            st.write("**Base Directory:**")
            st.code(BASE_DIR)
            
            st.write("**Database Path:**")
            st.code(DATABASE_PATH)
            st.write(f"Database exists: {os.path.exists(DATABASE_PATH)}")
            if os.path.exists(DATABASE_PATH):
                st.write(f"Database size: {os.path.getsize(DATABASE_PATH):,} bytes")
            
            st.write("**CSV Files Status:**")
            rga_files = ["rga_donors_2023", "rga_donors_2024"]
            
            for file_key in rga_files:
                file_path = CSV_FILES[file_key]
                exists = os.path.exists(file_path)
                if exists:
                    file_size = os.path.getsize(file_path)
                    st.success(f"✅ {file_key}: Found ({file_size:,} bytes)")
                    st.code(file_path)
                else:
                    st.error(f"❌ {file_key}: Not found")
                    st.code(file_path)
        
        if st.button("Check Database Status"):
            conn = get_connection()
            cursor = conn.cursor()
            
            st.write("**Database Tables:**")
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            rga_tables = [t for t in tables if 'rga' in t.lower()]
            if rga_tables:
                st.success(f"✅ RGA tables found: {rga_tables}")
                
                # Check RGA data counts
                for table in rga_tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        st.write(f"📊 {table}: {count} records")
                        
                        # Show sample data
                        cursor.execute(f"SELECT first_name, last_name, state, zip_code FROM {table} LIMIT 3")
                        samples = cursor.fetchall()
                        for sample in samples:
                            st.write(f"  • {sample[0]} {sample[1]} ({sample[2]}, {sample[3]})")
                    except Exception as e:
                        st.error(f"❌ Error querying {table}: {e}")
            else:
                st.error("❌ No RGA tables found!")
                st.write(f"Available tables: {tables}")
            
            conn.close()
        
        if st.button("Rebuild Database"):
            with st.spinner("Rebuilding database..."):
                try:
                    create_database()
                    st.success("✅ Database rebuilt successfully!")
                except Exception as e:
                    st.error(f"❌ Database rebuild failed: {e}")
                    st.exception(e)
        
        if st.button("Test RGA Matching"):
            from generic_processor import GenericDataProcessor
            import sys
            from io import StringIO
            
            # Capture print output
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                # Initialize processor with backward compatibility
                try:
                    processor = GenericDataProcessor(database_config=get_database_config())
                except TypeError:
                    processor = GenericDataProcessor()
                    if hasattr(processor, 'database_config'):
                        processor.database_config = get_database_config()
                
                # Test with known RGA donor
                test_donor = {
                    'first_name': 'MICHAEL',
                    'last_name': 'MANLOVE',
                    'state': 'AZ',
                    'zip_code': '85286',
                    'employer': ''
                }
                
                rga_flags = processor._check_rga_donors(test_donor)
                
                # Restore stdout
                sys.stdout = old_stdout
                
                # Show captured debug output
                debug_output = captured_output.getvalue()
                if debug_output:
                    st.text_area("Debug Output:", debug_output, height=100)
                
                if rga_flags:
                    st.success(f"✅ RGA matching works!")
                    st.write(f"Test result: {rga_flags[0]['source']}")
                else:
                    st.error("❌ RGA matching failed for test donor")
                    
            except Exception as e:
                sys.stdout = old_stdout
                st.error(f"Error during test: {e}")
        
        # Add a toggle for verbose debugging during uploads
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode for Uploads", value=False)
    
    # Sidebar for input options
    st.sidebar.header("Data Input Options")
    
    # STREAMLINED VERSION: Hide FEC API integration (preserved below for restoration)
    input_method = "Upload File"  # Force file upload only
    
    # ORIGINAL CODE (COMMENTED OUT FOR RESTORATION):
    # input_method = st.sidebar.radio(
    #     "Choose input method:",
    #     ["Upload File", "FEC API Search"]
    # )
    
    # Initialize processor with backward compatibility for database_config
    try:
        processor = FECDataProcessor(database_config=get_database_config())
    except TypeError:
        # Fallback for older version without database_config parameter
        processor = FECDataProcessor()
        # Set database_config manually if the processor has the attribute
        if hasattr(processor, 'database_config'):
            processor.database_config = get_database_config()
    
    if input_method == "Upload File":
        st.sidebar.subheader("📁 File Upload")
        
        # File format selection
        file_format = st.sidebar.radio(
            "Select file format:",
            ["FEC Quarterly Report", "State or Local Report"]
        )
        
        # Store file format in session state for summary page access
        st.session_state.file_format = file_format
        
        # Database selection options
        st.sidebar.markdown("---")
        st.sidebar.subheader("🗄️ Database Options")
        st.sidebar.markdown("Choose which databases to include in analysis:")
        
        # Initialize database settings in session state if not present
        if 'database_config' not in st.session_state:
            st.session_state['database_config'] = {
                'bad_donors': True,
                'rga_donors': True, 
                'bad_employers': True
            }
        
        # Database selection checkboxes
        db_config = st.session_state['database_config']
        
        db_config['bad_donors'] = st.sidebar.checkbox(
            "Bad Donor Master (Insurrectionists & Jan 6th)", 
            value=db_config['bad_donors'],
            key='cb_bad_donors'
        )
        
        db_config['rga_donors'] = st.sidebar.checkbox(
            "RGA Donors (2023 & 2024)", 
            value=db_config['rga_donors'],
            key='cb_rga_donors'
        )
        
        db_config['bad_employers'] = st.sidebar.checkbox(
            "Bad Employer Master", 
            value=db_config['bad_employers'],
            key='cb_bad_employers'
        )
        
        # Store updated config in session state
        st.session_state['database_config'] = db_config
        
        # Show instructions for State or Local Report
        if file_format == "State or Local Report":
            with st.sidebar.expander("📋 How to Use State/Local Reports", expanded=False):
                st.markdown("""
                **When to Use:**
                - State campaign finance data
                - Local election contribution files  
                - Any CSV with donor information (not FEC format)
                
                **File Requirements:**
                - CSV or Excel format
                - First row should contain column headers
                - Required: First Name, Last Name, State, Amount
                - Optional: ZIP Code, Date, Employer, Address
                
                **Step-by-Step:**
                1. **Upload your file** using the file uploader below
                2. **Skip rows** if your file has metadata/multiple headers
                3. **Map columns** by selecting which columns contain your data
                4. **Set focus state** (e.g., "NY") for ZIP code detail maps  
                5. **Click Analyze** to process and flag donors
                
                **Skip Rows:** Use if your file has:
                - Metadata at the top (agency info, report dates)
                - Multiple header rows
                - Blank rows before data starts
                
                **Amount Formats Supported:**
                - Plain numbers: `1000`, `250.50`
                - With dollar signs: `$1000`, `$250.50`
                - With commas: `1,000`, `$1,250.75`
                - Mixed formats in same file are OK
                
                **Focus State:** Enter 2-letter code (NY, CA, TX, etc.) to see detailed ZIP code maps for that state's contributors.
                """)
        
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV or Excel file containing donor data",
            type=['csv', 'xlsx', 'xls']
        )
        
        # Skip rows input for handling metadata/header rows
        skip_rows = st.sidebar.number_input(
            "Skip rows (metadata/headers):",
            min_value=0,
            max_value=20,
            value=0,
            help="Number of rows to skip at the beginning of the file (useful for files with metadata or multiple header rows)"
        )
        
        if uploaded_file:
            df = load_file_data(uploaded_file, skip_rows)
            if df is not None:
                st.subheader("📄 File Data Preview")
                st.write(f"Loaded {len(df)} rows, {len(df.columns)} columns from {uploaded_file.name}")
                st.dataframe(df.head(), use_container_width=True)
                
                if file_format == "State or Local Report":
                    st.subheader("🗂️ Column Mapping")
                    st.write("Map your CSV columns to the required fields:")
                    
                    # Get available columns
                    available_columns = ['(None)'] + list(df.columns)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Required Fields:**")
                        first_name_col = st.selectbox("First Name:", available_columns, key="first_name")
                        last_name_col = st.selectbox("Last Name:", available_columns, key="last_name")
                        state_col = st.selectbox("State:", available_columns, key="state")
                        amount_col = st.selectbox("Amount:", available_columns, key="amount")
                    
                    with col2:
                        st.write("**Optional Fields:**")
                        zip_col = st.selectbox("ZIP Code:", available_columns, key="zip_code")
                        date_col = st.selectbox("Date:", available_columns, key="date")
                        employer_col = st.selectbox("Employer:", available_columns, key="employer")
                        address_col = st.selectbox("Address:", available_columns, key="address")
                    
                    # Geographic focus state input
                    st.markdown("---")
                    st.write("**Geographic Analysis Settings:**")
                    geographic_focus_state = st.text_input(
                        "Focus State for ZIP Detail Map (e.g., NY, CA, TX):",
                        value="",
                        max_chars=2,
                        help="Enter a 2-letter state code to focus the ZIP code detail map. Leave blank to determine automatically from data.",
                        key="geographic_focus_state"
                    ).upper().strip()
                    
                    # Validate state code if provided
                    valid_states = {
                        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
                        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
                    }
                    
                    if geographic_focus_state and geographic_focus_state not in valid_states:
                        st.warning(f"⚠️ '{geographic_focus_state}' is not a valid US state code. Please use standard 2-letter codes like NY, CA, TX, etc.")
                    
                    # Target donation amount for max donors analysis
                    target_donation_amount = st.number_input(
                        "Target Donation Amount for Max Donors:",
                        value=2800,
                        min_value=1,
                        max_value=10000,
                        help="Donors within +/- $100 of this amount will be counted as 'Max Donors'",
                        key="target_donation_amount"
                    )
                    
                    # Validation
                    required_fields = [first_name_col, last_name_col, state_col, amount_col]
                    missing_required = [field for field in required_fields if field == '(None)']
                    
                    if missing_required:
                        st.warning(f"⚠️ Please map all required fields. Missing: {', '.join(['First Name', 'Last Name', 'State', 'Amount'][i] for i, field in enumerate(required_fields) if field == '(None)')}")
                        analyze_button_disabled = True
                    else:
                        st.success("✅ All required fields mapped!")
                        analyze_button_disabled = False
                        
                        # Store column mapping and geographic settings in session state
                        st.session_state['column_mapping'] = {
                            'first_name': first_name_col,
                            'last_name': last_name_col,
                            'state': state_col,
                            'amount': amount_col,
                            'zip_code': zip_col if zip_col != '(None)' else None,
                            'date': date_col if date_col != '(None)' else None,
                            'employer': employer_col if employer_col != '(None)' else None,
                            'address': address_col if address_col != '(None)' else None
                        }
                else:
                    analyze_button_disabled = False
                
                if st.button("🔍 Analyze Data", type="primary", disabled=analyze_button_disabled):
                    with st.spinner("Processing donor data and checking against databases..."):
                        try:
                            if file_format == "State or Local Report":
                                # Use generic processor for mapped columns
                                from generic_processor import GenericDataProcessor
                                # Initialize processor with backward compatibility
                                try:
                                    current_processor = GenericDataProcessor(database_config=get_database_config())
                                except TypeError:
                                    current_processor = GenericDataProcessor()
                                    if hasattr(current_processor, 'database_config'):
                                        current_processor.database_config = get_database_config()
                                
                                # Debug mode capture
                                if getattr(st.session_state, 'debug_mode', False):
                                    import sys
                                    from io import StringIO
                                    old_stdout = sys.stdout
                                    sys.stdout = captured_output = StringIO()
                                    
                                    try:
                                        results = current_processor.process_generic_data(df, st.session_state['column_mapping'])
                                        sys.stdout = old_stdout
                                        
                                        # Show debug output
                                        debug_output = captured_output.getvalue()
                                        if debug_output:
                                            st.expander("Debug Output", expanded=True).text_area("Processing Log:", debug_output, height=200)
                                    except Exception as e:
                                        sys.stdout = old_stdout
                                        raise e
                                else:
                                    results = current_processor.process_generic_data(df, st.session_state['column_mapping'])
                                
                                st.session_state['processor_type'] = 'generic'
                            else:
                                # Use FEC processor for standard format
                                current_processor = processor  # Use the global FEC processor
                                results = current_processor.process_fec_data(df)
                                st.session_state['processor_type'] = 'fec'
                            
                            # Store the processor instance for later use
                            st.session_state['current_processor'] = current_processor
                            
                            # Show processing results
                            bad_donors_count = len(results['bad_donors'])
                            
                            if bad_donors_count == 0:
                                st.warning("No individual donors found. Please check your data format and column mapping.")
                                # Show processing stats for debugging
                                if hasattr(current_processor, 'get_processing_stats'):
                                    stats = current_processor.get_processing_stats()
                                    st.info(f"**Processing Statistics:** {stats['processed_count']} processed, {stats['skipped_count']} skipped from {stats['total_rows']} total rows ({stats['success_rate']:.1f}% success rate)")
                            else:
                                st.session_state['results'] = results
                                st.session_state['original_df'] = df  # Store original DataFrame for geographic analysis
                                
                                # Show enhanced success message with processing stats
                                success_message = f"✅ Analysis complete! Found {bad_donors_count} individual donors."
                                if hasattr(current_processor, 'get_processing_stats'):
                                    stats = current_processor.get_processing_stats()
                                    success_message += f"\n📊 **Processing Stats:** {stats['processed_count']} processed, {stats['skipped_count']} skipped ({stats['success_rate']:.1f}% success rate)"
                                
                                st.success(success_message)
                                
                        except Exception as e:
                            st.error(f"Error processing data: {str(e)}")
                            st.info("Please check your file format and column mapping.")
    
    elif input_method == "FEC API Search":
        st.sidebar.subheader("🔎 FEC API Search")
        
        api_client = FECAPIClient()
        
        search_type = st.sidebar.selectbox(
            "Search for:",
            ["Candidates", "Committees"]
        )
        
        search_query = st.sidebar.text_input("Enter search term:")
        
        if search_query:
            if search_type == "Candidates":
                with st.spinner("Searching candidates..."):
                    candidates = api_client.search_candidates(search_query)
                
                if candidates:
                    st.subheader("Search Results - Candidates")
                    
                    for i, candidate in enumerate(candidates[:10]):  # Show top 10
                        with st.expander(f"{candidate['name']} ({candidate['party']}) - {candidate['office']} {candidate['state']}"):
                            st.write(f"**Candidate ID:** {candidate['candidate_id']}")
                            st.write(f"**Office:** {candidate['office']}")
                            st.write(f"**State:** {candidate['state']}")
                            if candidate['district']:
                                st.write(f"**District:** {candidate['district']}")
                            st.write(f"**Cycles:** {', '.join(map(str, candidate['cycles']))}")
                            
                            if st.button(f"Get Committees for {candidate['name']}", key=f"cand_{i}"):
                                committees = api_client.get_candidate_committees(candidate['candidate_id'])
                                if committees:
                                    st.write("**Associated Committees:**")
                                    for committee in committees:
                                        st.write(f"- {committee['name']} ({committee['committee_id']})")
            
            elif search_type == "Committees":
                with st.spinner("Searching committees..."):
                    committees = api_client.search_committees(search_query)
                
                if committees:
                    st.subheader("Search Results - Committees")
                    
                    selected_committee = st.selectbox(
                        "Select a committee to analyze:",
                        options=[(c['committee_id'], c['name']) for c in committees[:10]],
                        format_func=lambda x: f"{x[1]} ({x[0]})"
                    )
                    
                    if selected_committee:
                        committee_id = selected_committee[0]
                        committee_name = selected_committee[1]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            min_date = st.date_input("Start Date (optional)")
                        with col2:
                            max_date = st.date_input("End Date (optional)")
                        
                        limit = st.slider("Maximum records to fetch", 100, 5000, 1000)
                        
                        if st.button(f"📊 Analyze {committee_name}", type="primary"):
                            with st.spinner("Fetching data from FEC API..."):
                                df = api_client.get_schedule_a_data(
                                    committee_id=committee_id,
                                    min_date=min_date.strftime("%Y-%m-%d") if min_date else None,
                                    max_date=max_date.strftime("%Y-%m-%d") if max_date else None,
                                    limit=limit
                                )
                                
                                if not df.empty:
                                    st.success(f"Fetched {len(df)} records from FEC API")
                                    
                                    with st.spinner("Processing data and checking against databases..."):
                                        results = processor.process_fec_data(df)
                                        st.session_state['results'] = results
                                        st.session_state['original_df'] = df  # Store original DataFrame for geographic analysis
                                        st.success("Analysis complete!")
                                else:
                                    st.warning("No data found for the selected parameters.")
    
    # Display results if available
    if 'results' in st.session_state:
        st.markdown("---")
        display_results(st.session_state['results'])
    
    # Footer
    st.markdown("---")
    st.markdown("*FEC Donor & PAC Analysis Tool - Built for campaign finance transparency*")
    
    # Close the current processor if it exists
    if 'current_processor' in st.session_state:
        st.session_state['current_processor'].close()
    else:
        processor.close()  # Fallback to global processor

if __name__ == "__main__":
    main()