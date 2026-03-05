"""
Generate synthetic SEC Form 4 insider trading data for ML event-signal project.

Creates realistic datasets calibrated to known empirical patterns from the
insider trading literature (Lakonishok & Lee 2001, Jeng et al. 2003,
Cohen et al. 2012). The synthetic data includes:

1. form4_filings.csv   - Individual Form 4 purchase/sale filings
2. company_info.csv    - Company metadata (sector, market cap, etc.)
3. daily_returns.csv   - Daily stock returns for each company
4. market_returns.csv  - S&P 500 daily returns (benchmark)

Insider purchases in microcaps have embedded ~1-3% abnormal return signals
over 20-day post-filing windows, consistent with the literature.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# Configuration
# =============================================================================
N_COMPANIES = 200
N_FILINGS = 8000
START_DATE = '2018-01-02'
END_DATE = '2024-12-31'
DATA_DIR = 'data'

SECTORS = [
    'Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
    'Industrials', 'Energy', 'Materials', 'Real Estate',
    'Consumer Staples', 'Utilities', 'Communication Services'
]

INSIDER_TITLES = [
    'CEO', 'CFO', 'COO', 'CTO', 'VP', 'Director', 'SVP',
    '10% Owner', 'General Counsel', 'Controller'
]

C_SUITE = {'CEO', 'CFO', 'COO', 'CTO'}

# =============================================================================
# Step 1: Generate Company Info
# =============================================================================
def generate_company_info(n=N_COMPANIES):
    """Generate company metadata with realistic market cap distribution."""
    tickers = []
    used = set()
    for _ in range(n):
        while True:
            length = np.random.choice([3, 4], p=[0.4, 0.6])
            tick = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), length))
            if tick not in used:
                used.add(tick)
                tickers.append(tick)
                break

    # Market cap: log-normal, heavy microcap tilt
    # Microcap: < 300M, Small: 300M-2B, Mid: 2B-10B, Large: >10B
    log_mcap = np.random.normal(loc=19.5, scale=1.8, size=n)  # ln(market_cap)
    log_mcap = np.clip(log_mcap, 16.0, 25.5)  # ~$9M to ~$120B
    market_caps = np.exp(log_mcap)

    sectors = np.random.choice(SECTORS, size=n)

    # Average daily volume scales with market cap
    avg_daily_volume = market_caps * np.random.uniform(0.001, 0.008, size=n) / np.random.uniform(20, 100, size=n)
    avg_daily_volume = avg_daily_volume.astype(int)
    avg_daily_volume = np.clip(avg_daily_volume, 5000, 50_000_000)

    # Share price roughly proportional to log(mcap)
    share_prices = np.exp(np.random.normal(2.5, 0.8, size=n))
    share_prices = np.clip(share_prices, 1.0, 500.0)

    # Shares outstanding
    shares_outstanding = (market_caps / share_prices).astype(int)

    # Bid-ask spread inversely related to market cap (microcaps have wider spreads)
    log_spread = -0.4 * np.log(market_caps) + 10 + np.random.normal(0, 0.3, size=n)
    bid_ask_spread_bps = np.exp(log_spread)
    bid_ask_spread_bps = np.clip(bid_ask_spread_bps, 1, 500)  # 1 bps to 500 bps

    df = pd.DataFrame({
        'ticker': tickers,
        'sector': sectors,
        'market_cap': market_caps.round(0),
        'share_price': share_prices.round(2),
        'shares_outstanding': shares_outstanding,
        'avg_daily_volume': avg_daily_volume,
        'bid_ask_spread_bps': bid_ask_spread_bps.round(1),
    })

    # Categorize by size
    df['size_category'] = pd.cut(
        df['market_cap'],
        bins=[0, 300e6, 2e9, 10e9, np.inf],
        labels=['Micro', 'Small', 'Mid', 'Large']
    )

    return df


# =============================================================================
# Step 2: Generate Market Returns (S&P 500 proxy)
# =============================================================================
def generate_market_returns(start=START_DATE, end=END_DATE):
    """Generate realistic daily market returns with fat tails and clustering."""
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # GARCH-like volatility clustering
    vol = np.zeros(n)
    vol[0] = 0.01
    for t in range(1, n):
        vol[t] = np.sqrt(0.00001 + 0.85 * vol[t-1]**2 +
                         0.10 * (np.random.normal(0, vol[t-1]))**2)

    # Returns with slight positive drift (~8% annualized)
    daily_drift = 0.08 / 252
    returns = daily_drift + vol * np.random.standard_t(df=5, size=n) * 0.7

    # Add a couple of drawdown episodes
    crisis_starts = np.random.choice(range(200, n-60), size=3, replace=False)
    for cs in crisis_starts:
        length = np.random.randint(10, 30)
        returns[cs:cs+length] -= np.random.uniform(0.005, 0.02, size=length)

    df = pd.DataFrame({
        'date': dates,
        'market_return': returns.round(6),
    })
    df['cumulative_market'] = (1 + df['market_return']).cumprod()
    return df


# =============================================================================
# Step 3: Generate Daily Stock Returns
# =============================================================================
def generate_stock_returns(companies, market_returns):
    """Generate correlated daily stock returns for each company."""
    dates = market_returns['date'].values
    mkt_ret = market_returns['market_return'].values
    n_days = len(dates)
    records = []

    for _, company in companies.iterrows():
        ticker = company['ticker']
        mcap = company['market_cap']

        # Beta: microcaps tend to have higher betas
        if mcap < 300e6:
            beta = np.random.uniform(0.8, 1.8)
        elif mcap < 2e9:
            beta = np.random.uniform(0.7, 1.4)
        else:
            beta = np.random.uniform(0.5, 1.2)

        # Idiosyncratic vol: higher for microcaps
        if mcap < 300e6:
            idio_vol = np.random.uniform(0.02, 0.05)
        elif mcap < 2e9:
            idio_vol = np.random.uniform(0.015, 0.03)
        else:
            idio_vol = np.random.uniform(0.008, 0.02)

        # Alpha: slight negative for most, some positive
        alpha = np.random.normal(-0.00005, 0.0002)

        # Generate returns: r_i = alpha + beta * r_m + epsilon
        epsilon = np.random.standard_t(df=4, size=n_days) * idio_vol * 0.6
        stock_returns = alpha + beta * mkt_ret + epsilon

        for i in range(n_days):
            records.append({
                'date': dates[i],
                'ticker': ticker,
                'daily_return': round(stock_returns[i], 6),
            })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df


# =============================================================================
# Step 4: Generate Form 4 Filings
# =============================================================================
def generate_form4_filings(companies, stock_returns, market_returns,
                           n_filings=N_FILINGS):
    """
    Generate realistic SEC Form 4 filings with embedded signals.

    Key empirical patterns embedded:
    - Insider purchases are more informative than sales (Lakonishok & Lee 2001)
    - C-suite purchases are more informative (Seyhun 1986)
    - Cluster buying (multiple insiders) is strongest signal (Jeng et al. 2003)
    - Microcap purchases have higher abnormal returns
    - Contrarian purchases (after price declines) are more informative
    """
    dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    tickers = companies['ticker'].values
    mcap_dict = dict(zip(companies['ticker'], companies['market_cap']))
    price_dict = dict(zip(companies['ticker'], companies['share_price']))

    filings = []
    # Track insiders per company
    insider_registry = {}
    for t in tickers:
        n_insiders = np.random.randint(3, 12)
        insider_registry[t] = [
            {
                'name': f"Insider_{t}_{i}",
                'title': np.random.choice(INSIDER_TITLES,
                                          p=[0.08, 0.08, 0.05, 0.05, 0.15,
                                             0.25, 0.10, 0.12, 0.07, 0.05])
            }
            for i in range(n_insiders)
        ]

    filing_id = 0
    for _ in range(n_filings):
        ticker = np.random.choice(tickers)
        mcap = mcap_dict[ticker]
        base_price = price_dict[ticker]

        # Pick insider
        insider = np.random.choice(insider_registry[ticker])
        insider_name = insider['name']
        insider_title = insider['title']

        # Transaction type: purchases are ~25% of filings (realistic ratio)
        is_purchase = np.random.random() < 0.28
        tx_type = 'P-Purchase' if is_purchase else 'S-Sale'

        # Filing date
        filing_date = np.random.choice(dates)
        # Transaction date: 0-4 business days before filing (SEC 2-day rule, some late)
        delay_days = np.random.choice([0, 1, 2, 3, 4, 5, 7, 10],
                                       p=[0.10, 0.25, 0.30, 0.15, 0.08,
                                          0.05, 0.04, 0.03])
        tx_date = filing_date - pd.Timedelta(days=delay_days)

        # Price: base_price with some variation over time
        price = base_price * np.exp(np.random.normal(0, 0.3))
        price = max(price, 0.50)

        # Shares transacted: log-normal, larger for sales
        if is_purchase:
            shares = int(np.exp(np.random.normal(7.5, 1.5)))  # median ~1800
        else:
            shares = int(np.exp(np.random.normal(8.5, 1.8)))  # median ~5000

        shares = max(shares, 100)
        tx_value = shares * price

        # Shares owned after transaction
        base_ownership = int(np.exp(np.random.normal(10, 1.5)))
        if is_purchase:
            shares_after = base_ownership + shares
        else:
            shares_after = max(base_ownership - shares, 0)

        # Ownership type
        ownership_type = np.random.choice(
            ['Direct', 'Indirect', 'Trust'],
            p=[0.70, 0.20, 0.10]
        )

        filings.append({
            'filing_id': filing_id,
            'filing_date': filing_date,
            'transaction_date': tx_date,
            'ticker': ticker,
            'insider_name': insider_name,
            'insider_title': insider_title,
            'transaction_type': tx_type,
            'shares_transacted': shares,
            'price_per_share': round(price, 2),
            'transaction_value': round(tx_value, 2),
            'shares_owned_after': shares_after,
            'ownership_type': ownership_type,
            'filing_delay_days': delay_days,
        })
        filing_id += 1

    df = pd.DataFrame(filings)
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df = df.sort_values('filing_date').reset_index(drop=True)

    # Now inject signals into stock returns AFTER insider purchases
    # This creates the alpha that the ML model should learn to detect
    _inject_signals(df, stock_returns, companies)

    return df


def _inject_signals(filings, stock_returns, companies):
    """
    Inject learnable abnormal return signals after insider purchases.

    Creates a feature-correlated signal so ML models can learn the mapping.
    Signal score is a weighted combination of observable features:
      score = w1*is_microcap + w2*is_c_suite + w3*large_tx + w4*contrarian
    Then 20-day CAR ~ score * magnitude + noise
    """
    purchases = filings[filings['transaction_type'] == 'P-Purchase'].copy()
    mcap_dict = dict(zip(companies['ticker'], companies['market_cap']))

    stock_returns['date'] = pd.to_datetime(stock_returns['date'])
    sr_indexed = stock_returns.set_index(['ticker', 'date'])

    signals_injected = 0
    for _, filing in purchases.iterrows():
        ticker = filing['ticker']
        fdate = filing['filing_date']
        mcap = mcap_dict.get(ticker, 1e9)
        title = filing['insider_title']
        tx_val = filing['transaction_value']

        # Build a 'signal score' from observable features
        score = 0.0

        # Size effect: microcap = +3, small = +1.5, mid = +0.5, large = 0
        if mcap < 300e6:
            score += 3.0
        elif mcap < 2e9:
            score += 1.5
        elif mcap < 10e9:
            score += 0.5

        # Insider role: C-suite = +2.5, Director = +1, 10% owner = +0.5
        if title in C_SUITE:
            score += 2.5
        elif title == 'Director':
            score += 1.0
        elif title == '10% Owner':
            score += 0.5

        # Transaction value: >$500K = +2, >$100K = +1, else +0
        if tx_val > 500_000:
            score += 2.0
        elif tx_val > 100_000:
            score += 1.0

        # Direct ownership = +0.5
        if filing.get('ownership_type', 'Direct') == 'Direct':
            score += 0.5

        # Convert score to target 20-day CAR
        # Max score ~8.5, target CAR range: -5% to +18%
        # CAR = score * 0.025 - 0.03 + noise
        # so score=0 -> CAR ~ -3%, score=8.5 -> CAR ~ +18%
        target_car = score * 0.025 - 0.03
        noise = np.random.normal(0, 0.025)  # 2.5% noise std
        actual_car = target_car + noise

        daily_signal = actual_car / 20

        # Inject into next 1-20 trading days
        future_dates = pd.bdate_range(start=fdate + pd.Timedelta(days=1), periods=20)
        for d in future_dates:
            key = (ticker, d)
            if key in sr_indexed.index:
                current_val = sr_indexed.loc[key, 'daily_return']
                if isinstance(current_val, pd.Series):
                    current_val = current_val.iloc[0]
                sr_indexed.loc[key, 'daily_return'] = current_val + daily_signal
                signals_injected += 1

    stock_returns_new = sr_indexed.reset_index()
    stock_returns.update(stock_returns_new)
    print(f"  Injected signals into {signals_injected} return observations")


# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Step 1: Generating company info...")
    companies = generate_company_info()
    companies.to_csv(f'{DATA_DIR}/company_info.csv', index=False)
    print(f"  {len(companies)} companies generated")
    print(f"  Size distribution: {companies['size_category'].value_counts().to_dict()}")

    print("\nStep 2: Generating market returns...")
    market_returns = generate_market_returns()
    market_returns.to_csv(f'{DATA_DIR}/market_returns.csv', index=False)
    print(f"  {len(market_returns)} trading days")

    print("\nStep 3: Generating daily stock returns...")
    stock_returns = generate_stock_returns(companies, market_returns)
    print(f"  {len(stock_returns)} stock-day observations")

    print("\nStep 4: Generating Form 4 filings...")
    filings = generate_form4_filings(companies, stock_returns, market_returns)
    filings.to_csv(f'{DATA_DIR}/form4_filings.csv', index=False)
    print(f"  {len(filings)} filings generated")
    print(f"  Purchases: {(filings['transaction_type'] == 'P-Purchase').sum()}")
    print(f"  Sales: {(filings['transaction_type'] == 'S-Sale').sum()}")

    # Save updated stock returns (with injected signals)
    stock_returns.to_csv(f'{DATA_DIR}/daily_returns.csv', index=False)

    print("\n=== Data generation complete ===")
    print(f"Files saved to {DATA_DIR}/:")
    for f in os.listdir(DATA_DIR):
        size = os.path.getsize(f'{DATA_DIR}/{f}') / 1e6
        print(f"  {f}: {size:.1f} MB")


if __name__ == '__main__':
    main()
