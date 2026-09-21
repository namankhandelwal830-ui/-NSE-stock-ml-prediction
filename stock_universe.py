"""
stock_universe.py
------------------
A curated list of major Indian (NSE-listed) large-cap companies for the
dashboard's stock picker.

IMPORTANT HONESTY NOTE: NIFTY 50 constituents change periodically (index
rebalancing). This list is a manually curated snapshot of well-known,
liquid large-caps for convenience — it is NOT pulled live from NSE/NIFTY
indices (no free, reliable, scrape-safe API exists for that), so treat it
as "popular stocks", not an authoritative/current NIFTY 50 constituent list.
Users can always type any other NSE symbol directly.
"""

LAST_UPDATED = "2026-09-16"  # date this curated list was last manually reviewed

# (Symbol without .NS, Company Name, Sector)
STOCK_UNIVERSE = [
    ("RELIANCE", "Reliance Industries", "Energy/Conglomerate"),
    ("TCS", "Tata Consultancy Services", "IT Services"),
    ("HDFCBANK", "HDFC Bank", "Banking"),
    ("ICICIBANK", "ICICI Bank", "Banking"),
    ("INFY", "Infosys", "IT Services"),
    ("HINDUNILVR", "Hindustan Unilever", "FMCG"),
    ("ITC", "ITC Limited", "FMCG"),
    ("SBIN", "State Bank of India", "Banking"),
    ("BHARTIARTL", "Bharti Airtel", "Telecom"),
    ("KOTAKBANK", "Kotak Mahindra Bank", "Banking"),
    ("LT", "Larsen & Toubro", "Infrastructure/Construction"),
    ("AXISBANK", "Axis Bank", "Banking"),
    ("ASIANPAINT", "Asian Paints", "Consumer Goods"),
    ("MARUTI", "Maruti Suzuki India", "Automobile"),
    ("BAJFINANCE", "Bajaj Finance", "NBFC/Financial Services"),
    ("HCLTECH", "HCL Technologies", "IT Services"),
    ("WIPRO", "Wipro", "IT Services"),
    ("SUNPHARMA", "Sun Pharmaceutical", "Pharmaceuticals"),
    ("TITAN", "Titan Company", "Consumer Goods"),
    ("ULTRACEMCO", "UltraTech Cement", "Cement"),
    ("NESTLEIND", "Nestle India", "FMCG"),
    ("TATAMOTORS", "Tata Motors", "Automobile"),
    ("TATASTEEL", "Tata Steel", "Metals & Mining"),
    ("NTPC", "NTPC Limited", "Power"),
    ("POWERGRID", "Power Grid Corporation", "Power"),
    ("M&M", "Mahindra & Mahindra", "Automobile"),
    ("ADANIENT", "Adani Enterprises", "Conglomerate"),
    ("BAJAJFINSV", "Bajaj Finserv", "Financial Services"),
    ("ONGC", "Oil & Natural Gas Corporation", "Energy"),
    ("JSWSTEEL", "JSW Steel", "Metals & Mining"),
]


def get_display_options():
    """Returns display strings like 'RELIANCE — Reliance Industries' for the dropdown."""
    return [f"{sym} — {name}" for sym, name, _ in STOCK_UNIVERSE]


def parse_symbol_from_display(display_str: str) -> str:
    """Extracts just the NSE symbol from a display string like 'RELIANCE — Reliance Industries'."""
    return display_str.split(" — ")[0].strip()
