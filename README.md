# Tillennium Network

A staged company-intelligence research dashboard. This starter build includes:
- Streamlit dashboard and company/ticker search
- Yahoo Finance market and financial data (availability varies)
- Interactive Plotly charts
- Basic, transparent risk indicators
- Scenario calculator (illustrative assumptions)
- PDF snapshot reports

## Setup

1. Put your transparent white logo at `assets/tillennium_logo.png`.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended):
   - Windows: `python -m venv .venv` then `.venv\\Scripts\\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install: `python -m pip install -r requirements.txt`
5. Run: `streamlit run main.py`

If `python` does not work, try `python3`.

## Important data notes

Yahoo Finance data may be delayed, incomplete, unavailable, or subject to provider terms.
The app labels its source and retrieval time. It does not claim an exchange-certified live feed.
Risk indicators are descriptive heuristics, not investment advice or forecasts.
SEC, supply-chain and geopolitical modules are clearly marked as not yet connected.
Portfolio Analysis means analysis of publicly disclosed institutional ownership only;
it does not manage portfolios or execute trades.

## Logo

Use a transparent PNG named exactly `tillennium_logo.png` inside `assets/`.
The supplied white logo is displayed on dark backgrounds. A white-background PDF
will use a dark logo panel so the white lettering remains visible.


## Streamlit Community Cloud deployment

The Python modules and asset files are kept in the project root (there are no
`modules/` or `assets/` subfolders). Keep `main.py`, `requirements.txt`, the
Python module files, logo images, and `styles.css` together in the repository
root. The `reports/` folder is retained for generated PDF reports.

To deploy:
1. Push the project files to a GitHub repository.
2. In Streamlit Community Cloud, select that repository.
3. Set the app file path to `main.py`.
