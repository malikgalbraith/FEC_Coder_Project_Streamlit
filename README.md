# FEC Donor & PAC Analysis Tool

A comprehensive Streamlit web application for analyzing Federal Election Commission (FEC) data, including donor information, PAC contributions, and geographic visualizations.

## Features

- **Real-time FEC Data Integration**: Fetch live data from the FEC API
- **Comprehensive Database**: Local SQLite database with master lists for:
  - Bad actors and problematic donors
  - Industry classifications
  - Committee information
  - Geographic data with ZIP code coordinates
- **Interactive Visualizations**: 
  - Plotly charts and graphs
  - Interactive maps with Folium
  - Geographic analysis and choropleth maps
- **Multi-format Export**: Download results in CSV, Excel, and other formats
- **Advanced Filtering**: Search and filter by multiple criteria

## Deployment

This application is configured for deployment on Streamlit Community Cloud.

### Requirements

- Python 3.8+
- Streamlit
- FEC API Key (configured via Streamlit secrets)

### Local Development

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your FEC API key in `.streamlit/secrets.toml`
4. Run the app: `streamlit run app.py`

## Configuration

The app requires an FEC API key for accessing live data. In production, this should be configured through Streamlit Cloud's secrets management.

## License

This project is designed for research and educational purposes in campaign finance analysis.