"""
Pull real SEC Form 4 insider trading data from WRDS for the ML event-signal project.

Sources:
  - TFN table1:  Non-derivative insider transactions (Form 4 purchases)
  - CRSP dsf:    Daily stock returns, price, volume, shares outstanding
  - CRSP dsenames: Company metadata (name, SIC, exchange)
  - FF factors:  Fama-French daily factors (market return benchmark)

Requires: wrds package with valid credentials (pgpass configured).
"""

import wrds
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

WRDS_USER = 'zaranip'
START_DATE = '2018-01-01'
END_DATE = '2024-12-31'

# =========================================================================
# Connect
# =========================================================================
print("Connecting to WRDS...")
db = wrds.Connection(wrds_username=WRDS_USER)
print("Connected.\n")

# =========================================================================
# 1. Pull TFN Form 4 insider transactions (open-market purchases + sales)
# =========================================================================
print("=" * 70)
print("Step 1: Pulling TFN insider transactions...")
print("=" * 70)

tfn_query = f"""
SELECT
    t.fdate,
    t.trandate,
    t.ticker,
    t.personid,
    t.owner,
    t.cname,
    t.rolecode1,
    t.rolecode2,
    t.trancode,
    t.acqdisp,
    t.tprice,
    t.shares,
    t.sharesheld,
    t.ownership,
    t.cleanse,
    t.sector,
    t.industry,
    t.cusip6,
    t.formtype
FROM tfn.table1 t
WHERE t.fdate BETWEEN '{START_DATE}' AND '{END_DATE}'
  AND t.formtype IN ('4', '4/A')
  AND t.trancode IN ('P', 'S')
  AND t.cleanse IN ('R', 'S', 'A')
  AND t.tprice > 0
  AND t.shares > 0
  AND t.ticker IS NOT NULL
  AND LENGTH(TRIM(t.ticker)) > 0
ORDER BY t.fdate, t.ticker
"""

tfn_raw = db.raw_sql(tfn_query)
print(f"  Raw TFN rows: {len(tfn_raw):,}")

# Clean up
tfn_raw['fdate'] = pd.to_datetime(tfn_raw['fdate'])
tfn_raw['trandate'] = pd.to_datetime(tfn_raw['trandate'])
tfn_raw['ticker'] = tfn_raw['ticker'].str.strip().str.upper()
tfn_raw['transaction_type'] = tfn_raw['acqdisp'].map({'A': 'P-Purchase', 'D': 'S-Sale'})
tfn_raw['transaction_value'] = tfn_raw['tprice'] * tfn_raw['shares']

# Map role codes to readable titles
ROLE_MAP = {
    'CEO': 'CEO', 'CFO': 'CFO', 'CO': 'COO', 'CT': 'CTO',
    'CB': 'Chairman', 'P': 'President', 'D': 'Director',
    'DO': 'Director/Officer', 'VP': 'VP', 'SVP': 'SVP', 'EVP': 'EVP',
    'GC': 'General Counsel', 'H': 'Officer', 'O': 'Officer',
    'OB': 'Board Member', 'OD': 'Director', 'OX': 'Other Executive',
    'T': 'Treasurer', 'VC': 'Vice Chair', 'GP': 'General Partner',
    'X': 'Other', 'AV': 'VP',
}
tfn_raw['insider_title'] = tfn_raw['rolecode1'].map(ROLE_MAP).fillna('Other')

# Filing delay
tfn_raw['filing_delay_days'] = (tfn_raw['fdate'] - tfn_raw['trandate']).dt.days
tfn_raw['filing_delay_days'] = tfn_raw['filing_delay_days'].clip(lower=0, upper=60)

# Ownership type
tfn_raw['ownership_type'] = tfn_raw['ownership'].map({'D': 'Direct', 'I': 'Indirect'}).fillna('Other')

# Build final Form 4 dataframe
form4 = tfn_raw.rename(columns={
    'fdate': 'filing_date',
    'trandate': 'transaction_date',
    'owner': 'insider_name',
    'tprice': 'price_per_share',
    'shares': 'shares_transacted',
    'sharesheld': 'shares_owned_after',
}).copy()

form4['filing_id'] = range(len(form4))

keep_cols = [
    'filing_id', 'filing_date', 'transaction_date', 'ticker',
    'insider_name', 'insider_title', 'transaction_type',
    'shares_transacted', 'price_per_share', 'transaction_value',
    'shares_owned_after', 'ownership_type', 'filing_delay_days',
    'cusip6', 'cname', 'personid', 'rolecode1', 'sector', 'industry'
]
form4 = form4[[c for c in keep_cols if c in form4.columns]]

# Drop obvious bad data
form4 = form4[form4['price_per_share'] > 0.5]  # penny stocks out
form4 = form4[form4['price_per_share'] < 10000]  # outliers
form4 = form4[form4['shares_transacted'] > 0]
form4 = form4.dropna(subset=['filing_date', 'ticker'])

print(f"  Cleaned Form 4 rows: {len(form4):,}")
print(f"  Purchases: {(form4['transaction_type'] == 'P-Purchase').sum():,}")
print(f"  Sales: {(form4['transaction_type'] == 'S-Sale').sum():,}")
print(f"  Unique tickers: {form4['ticker'].nunique():,}")
print(f"  Date range: {form4['filing_date'].min().date()} to {form4['filing_date'].max().date()}")

form4.to_csv(f'{DATA_DIR}/form4_filings.csv', index=False)
print(f"  Saved to {DATA_DIR}/form4_filings.csv")

# =========================================================================
# 2. Get unique tickers → pull CRSP identifiers
# =========================================================================
print("\n" + "=" * 70)
print("Step 2: Matching tickers to CRSP permnos...")
print("=" * 70)

unique_tickers = form4['ticker'].unique().tolist()
print(f"  Unique tickers to match: {len(unique_tickers)}")

# Use CRSP dsenames to get permno for each ticker
# Pull all CRSP name records and match on ticker
ticker_str = "','".join(unique_tickers[:5000])  # SQL limit safety

crsp_names_query = f"""
SELECT DISTINCT permno, ticker, comnam, siccd, exchcd, shrcd,
       namedt, nameendt, cusip
FROM crsp_a_stock.dsenames
WHERE ticker IN ('{ticker_str}')
  AND shrcd IN (10, 11)
  AND namedt <= '{END_DATE}'
  AND nameendt >= '{START_DATE}'
"""

crsp_names = db.raw_sql(crsp_names_query)
crsp_names['ticker'] = crsp_names['ticker'].str.strip().str.upper()
print(f"  CRSP name records: {len(crsp_names):,}")
print(f"  Unique permnos matched: {crsp_names['permno'].nunique():,}")

# Keep the most recent name record per permno-ticker pair
crsp_names = (crsp_names.sort_values('nameendt', ascending=False)
              .drop_duplicates(subset=['permno', 'ticker'], keep='first'))

# Map tickers to permnos (take the first match if multiple)
ticker_permno = (crsp_names.drop_duplicates(subset='ticker', keep='first')
                 [['ticker', 'permno', 'comnam', 'siccd', 'exchcd']])
print(f"  Ticker-permno mappings: {len(ticker_permno):,}")

# =========================================================================
# 3. Pull CRSP daily stock data
# =========================================================================
print("\n" + "=" * 70)
print("Step 3: Pulling CRSP daily returns...")
print("=" * 70)

matched_permnos = ticker_permno['permno'].unique().tolist()
print(f"  Pulling daily data for {len(matched_permnos):,} permnos...")

# Pull in chunks to avoid query size limits
chunk_size = 500
all_dsf = []

for i in range(0, len(matched_permnos), chunk_size):
    chunk = matched_permnos[i:i+chunk_size]
    permno_str = ','.join(str(int(p)) for p in chunk)

    dsf_query = f"""
    SELECT permno, date, ret, prc, shrout, vol
    FROM crsp_a_stock.dsf
    WHERE permno IN ({permno_str})
      AND date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    chunk_df = db.raw_sql(dsf_query)
    all_dsf.append(chunk_df)
    print(f"    Chunk {i//chunk_size + 1}: {len(chunk_df):,} rows")

dsf = pd.concat(all_dsf, ignore_index=True)
dsf['date'] = pd.to_datetime(dsf['date'])
print(f"  Total CRSP daily rows: {len(dsf):,}")

# Merge permno → ticker
dsf = dsf.merge(ticker_permno[['permno', 'ticker']], on='permno', how='left')

# Compute market cap: abs(prc) * shrout * 1000 (shrout is in thousands)
dsf['market_cap'] = dsf['prc'].abs() * dsf['shrout'] * 1000
dsf['avg_dollar_volume'] = dsf['prc'].abs() * dsf['vol']

# Rename for consistency
daily_returns = dsf.rename(columns={'ret': 'daily_return'})[
    ['date', 'ticker', 'permno', 'daily_return', 'prc', 'shrout', 'vol',
     'market_cap', 'avg_dollar_volume']
].copy()

# Drop missing returns
daily_returns = daily_returns.dropna(subset=['daily_return'])
print(f"  Clean daily return rows: {len(daily_returns):,}")

daily_returns.to_csv(f'{DATA_DIR}/daily_returns.csv', index=False)
print(f"  Saved to {DATA_DIR}/daily_returns.csv")

# =========================================================================
# 4. Build company info from CRSP
# =========================================================================
print("\n" + "=" * 70)
print("Step 4: Building company info...")
print("=" * 70)

# Most recent snapshot for each ticker
latest = (daily_returns.sort_values('date')
          .drop_duplicates(subset='ticker', keep='last'))

company_info = latest[['ticker', 'permno', 'market_cap']].copy()
company_info = company_info.merge(
    ticker_permno[['ticker', 'comnam', 'siccd', 'exchcd']],
    on='ticker', how='left'
)

# Average daily volume over the full period
avg_vol = (daily_returns.groupby('ticker')['vol']
           .mean().reset_index()
           .rename(columns={'vol': 'avg_daily_volume'}))
company_info = company_info.merge(avg_vol, on='ticker', how='left')

# Average dollar volume
avg_dvol = (daily_returns.groupby('ticker')['avg_dollar_volume']
            .mean().reset_index()
            .rename(columns={'avg_dollar_volume': 'avg_daily_dollar_volume'}))
company_info = company_info.merge(avg_dvol, on='ticker', how='left')

# Average share price
avg_prc = (daily_returns.groupby('ticker')['prc']
           .apply(lambda x: x.abs().mean()).reset_index()
           .rename(columns={'prc': 'avg_share_price'}))
company_info = company_info.merge(avg_prc, on='ticker', how='left')

# Size category
company_info['size_category'] = pd.cut(
    company_info['market_cap'],
    bins=[0, 300e6, 2e9, 10e9, np.inf],
    labels=['Micro', 'Small', 'Mid', 'Large']
)

# Estimate bid-ask spread from daily return volatility and volume
# Corwin-Schultz (2012) proxy: higher vol + lower volume → wider spread
daily_vol = (daily_returns.groupby('ticker')['daily_return']
             .std().reset_index()
             .rename(columns={'daily_return': 'return_volatility'}))
company_info = company_info.merge(daily_vol, on='ticker', how='left')

# Rough spread estimate: 2 * vol / sqrt(volume) * 10000 bps
company_info['bid_ask_spread_bps'] = (
    2 * company_info['return_volatility'] /
    np.sqrt(company_info['avg_daily_volume'].clip(lower=100)) * 10000
).clip(lower=1, upper=500)

# Map SIC codes to broad sectors
def sic_to_sector(sic):
    if pd.isna(sic):
        return 'Other'
    sic = int(sic)
    if 100 <= sic <= 999:
        return 'Agriculture'
    elif 1000 <= sic <= 1499:
        return 'Mining'
    elif 1500 <= sic <= 1799:
        return 'Construction'
    elif 2000 <= sic <= 3999:
        return 'Manufacturing'
    elif 4000 <= sic <= 4999:
        return 'Utilities/Transport'
    elif 5000 <= sic <= 5199:
        return 'Wholesale'
    elif 5200 <= sic <= 5999:
        return 'Retail'
    elif 6000 <= sic <= 6799:
        return 'Finance'
    elif 7000 <= sic <= 8999:
        return 'Services'
    elif 9000 <= sic <= 9999:
        return 'Public Admin'
    else:
        return 'Other'

company_info['sector'] = company_info['siccd'].apply(sic_to_sector)

print(f"  Companies: {len(company_info):,}")
print(f"  Size distribution:\n{company_info['size_category'].value_counts()}")
print(f"  Sector distribution:\n{company_info['sector'].value_counts().head(10)}")

company_info.to_csv(f'{DATA_DIR}/company_info.csv', index=False)
print(f"  Saved to {DATA_DIR}/company_info.csv")

# =========================================================================
# 5. Pull Fama-French daily factors (market return benchmark)
# =========================================================================
print("\n" + "=" * 70)
print("Step 5: Pulling Fama-French daily factors...")
print("=" * 70)

ff_query = f"""
SELECT date, mktrf, smb, hml, rf
FROM ff.factors_daily
WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
ORDER BY date
"""

ff = db.raw_sql(ff_query)
ff['date'] = pd.to_datetime(ff['date'])

# Market return = mktrf + rf
ff['market_return'] = ff['mktrf'] + ff['rf']
ff['cumulative_market'] = (1 + ff['market_return']).cumprod()

print(f"  FF factor rows: {len(ff):,}")
print(f"  Date range: {ff['date'].min().date()} to {ff['date'].max().date()}")

market_returns = ff[['date', 'market_return', 'mktrf', 'smb', 'hml', 'rf', 'cumulative_market']]
market_returns.to_csv(f'{DATA_DIR}/market_returns.csv', index=False)
print(f"  Saved to {DATA_DIR}/market_returns.csv")

# =========================================================================
# Close connection
# =========================================================================
db.close()
print("\n" + "=" * 70)
print("WRDS connection closed.")
print("=" * 70)

# =========================================================================
# Summary
# =========================================================================
print("\n=== Data Pull Summary ===")
for fname in os.listdir(DATA_DIR):
    fpath = f'{DATA_DIR}/{fname}'
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"  {fname}: {size_mb:.1f} MB")

print(f"\nForm 4 filings: {len(form4):,}")
print(f"  Purchases: {(form4['transaction_type'] == 'P-Purchase').sum():,}")
print(f"  Sales: {(form4['transaction_type'] == 'S-Sale').sum():,}")
print(f"  Tickers: {form4['ticker'].nunique():,}")
print(f"Daily returns: {len(daily_returns):,} rows")
print(f"Companies: {len(company_info):,}")
print(f"Market return days: {len(market_returns):,}")
print("\nDone.")
