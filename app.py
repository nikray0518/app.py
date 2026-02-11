import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
from datetime import datetime, timedelta
import math
import requests
import warnings
warnings.filterwarnings('ignore')

# --- UTILS ---
# Replacement for scipy.stats.norm to remove heavy dependency
class NormalDistribution:
    def cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    def pdf(self, x):
        return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)

norm = NormalDistribution()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Options.AI | Credit Spread Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR INSTITUTIONAL UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    /* Root Variables */
    :root {
        --bg-primary: #1d1f21; 
        --bg-secondary: #282a2e;
        --bg-card: #373b41;
        --bg-elevated: #454a52;
        --border-color: #5c6168;
        --text-primary: #ffffff;
        --text-secondary: #c5c8c6;
        --text-muted: #969896;
        --accent-green: #00ff7f;
        --accent-green-dim: rgba(0, 255, 127, 0.15);
        --accent-red: #ff5555;
        --accent-red-dim: rgba(255, 85, 85, 0.15);
        --accent-blue: #81a2be;
        --accent-purple: #b294bb;
        --accent-gold: #f0c674;
    }
    
    /* Global Styles */
    body, .stApp {
        font-family: 'Roboto', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main .block-container {
        padding: 1.5rem 2rem; /* Reduced padding */
        max-width: 1400px;   /* Slightly smaller max-width */
    }

    /* New Content Card style */
    .content-card {
        background-color: var(--bg-secondary);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
    }
    
    /* Hero Title */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #a0a0b0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    /* Premium Cards */
    .metric-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.25rem;
        min-height: 90px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(129, 162, 190, 0.1);
    }
    
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .metric-value-green {
        color: var(--accent-green) !important;
    }
    
    .metric-value-red {
        color: var(--accent-red) !important;
    }
    
    /* Stock Price Display */
    .stock-price-container {
        background: linear-gradient(135deg, var(--bg-elevated) 0%, var(--bg-card) 100%);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .stock-price-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-green), var(--accent-blue), var(--accent-purple));
    }
    
    .stock-symbol {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: 0.1em;
    }
    
    .stock-price {
        font-size: 3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0.5rem 0;
    }
    
    /* Streamlit Overrides */
    .stTextInput > div > div > input {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(129, 162, 190, 0.2) !important;
    }
    
    .stSelectbox > div > div {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
    }
    
    .stRadio > div {
        gap: 1rem !important;
    }
    
    .stRadio > div > label {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stRadio > div > label:hover {
        border-color: var(--accent-blue) !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        border-color: var(--accent-green) !important;
        background: var(--accent-green-dim) !important;
    }
    
    .stButton > button {
        background-color: #4285F4 !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important; /* Reduced padding */
        font-size: 0.9rem !important; /* Smaller font size */
        line-height: 1.5 !important; /* Ensure consistent height */
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #3367D6 !important; /* A darker shade for hover */
        box-shadow: 0 2px 3px rgba(0, 0, 0, 0.1) !important;
        transform: none !important;
    }
    
    /* Number Input */
    .stNumberInput > div > div > input {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    
    /* Divider */
    hr {
        border-color: var(--border-color) !important;
        margin: 2rem 0 !important;
    }
    
    /* Tab styling - More prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-secondary);
        padding: 0.75rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 1.1rem !important;
        padding: 1rem 2rem !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: var(--bg-card);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(129, 162, 190, 0.3) 0%, rgba(129, 162, 190, 0.15) 100%) !important;
        border: 1px solid rgba(129, 162, 190, 0.5) !important;
        color: var(--accent-blue) !important;
        font-weight: 700 !important;
    }
    
    /* Tab panel */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }
    
    /* Live indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        background: var(--accent-green-dim);
        border: 1px solid var(--accent-green);
        border-radius: 50px;
        font-size: 0.75rem;
        color: var(--accent-green);
        font-weight: 600;
    }
    
    .live-dot {
        width: 6px;
        height: 6px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Greek styling */
    .greek-item {
        background: var(--bg-elevated);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .greek-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .greek-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 0.25rem;
    }

    /* Buy Me Coffee Button */
    .bmc-button {
        position: fixed;
        top: 1.5rem;
        right: 2rem;
        z-index: 999999;
        background-color: #FFDD00;
        color: #000000 !important;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        text-decoration: none !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        border: 1px solid #e6c300;
    }
    
    .bmc-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
        background-color: #ffe64a;
    }
</style>
<a href="https://www.buymeacoffee.com" target="_blank" class="bmc-button">☕ Buy me a Coffee</a>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 CORE CALCULATIONS
# ==========================================

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        if option_type == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    if option_type == 'call':
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def calculate_spread_greeks(S, short_K, long_K, T, r, sigma_short, sigma_long, spread_type):
    if spread_type == "Bull Put":
        short_greeks = calculate_greeks(S, short_K, T, r, sigma_short, 'put')
        long_greeks = calculate_greeks(S, long_K, T, r, sigma_long, 'put')
        net_delta = -short_greeks['delta'] + long_greeks['delta']
        net_gamma = -short_greeks['gamma'] + long_greeks['gamma']
        net_theta = -short_greeks['theta'] + long_greeks['theta']
        net_vega = -short_greeks['vega'] + long_greeks['vega']
    else:
        short_greeks = calculate_greeks(S, short_K, T, r, sigma_short, 'call')
        long_greeks = calculate_greeks(S, long_K, T, r, sigma_long, 'call')
        net_delta = -short_greeks['delta'] + long_greeks['delta']
        net_gamma = -short_greeks['gamma'] + long_greeks['gamma']
        net_theta = -short_greeks['theta'] + long_greeks['theta']
        net_vega = -short_greeks['vega'] + long_greeks['vega']
    
    return {
        'delta': net_delta,
        'gamma': net_gamma,
        'theta': net_theta,
        'vega': net_vega
    }

def calculate_probability_of_profit(S, K, T, r, sigma, spread_type, net_credit, width):
    if T <= 0 or sigma <= 0:
        return 50.0
    
    if spread_type == "Bull Put":
        breakeven = K - (net_credit / 100)
        d2 = (np.log(S / breakeven) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        pop = norm.cdf(d2) * 100
    else:
        breakeven = K + (net_credit / 100)
        d2 = (np.log(S / breakeven) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        pop = norm.cdf(-d2) * 100
    
    pop = max(5.0, min(95.0, pop))
    return pop

def calculate_expected_value(max_profit, max_loss, prob_profit):
    prob_win = prob_profit / 100
    prob_loss = 1 - prob_win
    ev = (prob_win * max_profit) - (prob_loss * max_loss)
    return ev

def calculate_kelly_criterion(prob_profit, max_profit, max_loss):
    if max_loss <= 0:
        return 0
    
    b = max_profit / max_loss
    p = prob_profit / 100
    q = 1 - p
    
    kelly = (b * p - q) / b
    return max(0, min(kelly, 0.25)) * 100

def calculate_iv_rank(current_iv, iv_history):
    if len(iv_history) == 0:
        return 50.0
    
    iv_low = np.min(iv_history)
    iv_high = np.max(iv_history)
    
    if iv_high == iv_low:
        return 50.0
    
    iv_rank = ((current_iv - iv_low) / (iv_high - iv_low)) * 100
    return iv_rank

def run_monte_carlo_gbm(S, r, sigma, T, n_sims=10000, n_steps=None):
    if n_steps is None:
        n_steps = max(int(T * 252), 10)
    
    dt = T / n_steps
    Z = np.random.standard_normal((n_steps, n_sims))
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion
    log_prices = np.cumsum(log_returns, axis=0)
    price_paths = S * np.exp(np.vstack([np.zeros(n_sims), log_prices]))
    
    return price_paths

def evaluate_spread_monte_carlo(S, short_K, long_K, T, r, sigma, spread_type, net_credit, quantity, n_sims=10000):
    paths = run_monte_carlo_gbm(S, r, sigma, T, n_sims)
    final_prices = paths[-1]
    
    max_loss = (abs(short_K - long_K) * 100 * quantity) - net_credit
    
    if spread_type == "Bull Put":
        pnl = np.where(
            final_prices >= short_K,
            net_credit,
            np.where(
                final_prices <= long_K,
                -max_loss,
                net_credit - ((short_K - final_prices) * 100 * quantity)
            )
        )
        wins = np.sum(final_prices > (short_K - net_credit / (100 * quantity)))
    else:
        pnl = np.where(
            final_prices <= short_K,
            net_credit,
            np.where(
                final_prices >= long_K,
                -max_loss,
                net_credit - ((final_prices - short_K) * 100 * quantity)
            )
        )
        wins = np.sum(final_prices < (short_K + net_credit / (100 * quantity)))
    
    win_rate = (wins / n_sims) * 100
    expected_pnl = np.mean(pnl)
    
    return {
        'win_rate': win_rate,
        'expected_pnl': expected_pnl,
        'pnl_distribution': pnl,
        'final_prices': final_prices,
        'paths': paths,
        'percentile_5': np.percentile(pnl, 5),
        'percentile_95': np.percentile(pnl, 95),
        'median_pnl': np.median(pnl),
        'std_dev': np.std(pnl)
    }

def generate_pnl_chart(short_strike, long_strike, net_premium, quantity, spread_type, current_price):
    max_loss = (abs(short_strike - long_strike) * 100 * quantity) - net_premium
    
    lower_bound = min(short_strike, long_strike) * 0.88
    upper_bound = max(short_strike, long_strike) * 1.12
    price_range = np.linspace(lower_bound, upper_bound, 200)
    
    if spread_type == "Bull Put":
        pnl = np.where(
            price_range >= short_strike,
            net_premium,
            np.where(
                price_range <= long_strike,
                -max_loss,
                net_premium - ((short_strike - price_range) * 100 * quantity)
            )
        )
        breakeven = short_strike - (net_premium / (100 * quantity))
    else:
        pnl = np.where(
            price_range <= short_strike,
            net_premium,
            np.where(
                price_range >= long_strike,
                -max_loss,
                net_premium - ((price_range - short_strike) * 100 * quantity)
            )
        )
        breakeven = short_strike + (net_premium / (100 * quantity))
    
    chart_data = pd.DataFrame({
        'Price': price_range,
        'P/L': pnl
    })
    
    line = alt.Chart(chart_data).mark_line(
        strokeWidth=3,
        color='#81a2be'
    ).encode(
        x=alt.X('Price:Q', title='Stock Price at Expiration', axis=alt.Axis(format='$,.0f')),
        y=alt.Y('P/L:Q', title='Profit / Loss', axis=alt.Axis(format='$,.0f'))
    )
    
    profit_data = chart_data[chart_data['P/L'] >= 0].copy()
    profit_area = alt.Chart(profit_data).mark_area(
        opacity=0.3,
        color='#00ff7f'
    ).encode(
        x='Price:Q',
        y='P/L:Q',
        y2=alt.datum(0)
    )
    
    loss_data = chart_data[chart_data['P/L'] < 0].copy()
    loss_area = alt.Chart(loss_data).mark_area(
        opacity=0.3,
        color='#ff5555'
    ).encode(
        x='Price:Q',
        y='P/L:Q',
        y2=alt.datum(0)
    )
    
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='#5c6168',
        strokeDash=[4, 4]
    ).encode(y='y:Q')
    
    current_line = alt.Chart(pd.DataFrame({'x': [current_price]})).mark_rule(
        color='#81a2be',
        strokeWidth=2,
        strokeDash=[6, 3]
    ).encode(x='x:Q')
    
    breakeven_line = alt.Chart(pd.DataFrame({'x': [breakeven]})).mark_rule(
        color='#f0c674',
        strokeWidth=2,
        strokeDash=[4, 4]
    ).encode(x='x:Q')
    
    chart = (profit_area + loss_area + zero_line + breakeven_line + current_line + line).properties(
        height=350
    ).configure_axis(
        labelColor='#969896',
        titleColor='#969896',
        gridColor='#373b41',
        domainColor='#373b41'
    ).configure_view(
        strokeWidth=0
    )
    
    return chart, breakeven, max_loss

def calculate_score(roi, pop, ev, kelly, theta):
    roi_score = min(roi / 50, 1) * 25
    pop_score = (pop / 100) * 30
    ev_score = min(max(ev + 50, 0) / 100, 1) * 25
    kelly_score = min(kelly / 10, 1) * 10
    theta_score = min(max(theta, 0) / 10, 1) * 10
    
    return roi_score + pop_score + ev_score + kelly_score + theta_score

def get_news_sentiment(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news: return 0
        
        sentiment_score = 0
        count = 0
        
        # AI Lexicon for Sentiment Analysis
        bullish_tags = ['upgrade', 'buy', 'outperform', 'strong', 'growth', 'gain', 'surge', 'jump', 'rally', 'beat', 'positive', 'high', 'soar', 'bull', 'revenue', 'profit']
        bearish_tags = ['downgrade', 'sell', 'underperform', 'weak', 'miss', 'loss', 'drop', 'fall', 'decline', 'plunge', 'crash', 'negative', 'low', 'bear', 'inflation', 'concern']
        
        for article in news[:10]:
            title = article.get('title', '').lower()
            score = 0
            if any(x in title for x in bullish_tags): score += 1
            if any(x in title for x in bearish_tags): score -= 1
            
            if score > 0: sentiment_score += 1
            elif score < 0: sentiment_score -= 1
            count += 1
            
        if count == 0: return 0
        
        final_score = sentiment_score / count
        if final_score > 0.15: return 1
        elif final_score < -0.15: return -1
        else: return 0
    except:
        return 0

def calculate_ai_rating(score, pop, ev, sentiment_val=0, spread_type="Bull Put Spread"):
    adjusted_score = score
    is_bullish = "Bull" in spread_type
    
    if is_bullish:
        if sentiment_val == 1: adjusted_score += 10
        elif sentiment_val == -1: adjusted_score -= 15
    else:
        if sentiment_val == -1: adjusted_score += 10
        elif sentiment_val == 1: adjusted_score -= 15
        
    if adjusted_score >= 70 and pop > 60 and ev > 0:
        return "Great"
    elif adjusted_score >= 50 and pop > 50:
        return "Good"
    elif adjusted_score >= 30:
        return "Fair"
    else:
        return "Bad"

def scan_spread_opportunities(opt_chain, current_price, T_years, hist_volatility, risk_free_rate, spread_type, sentiment_val=0):
    spreads = []
    if spread_type == "Bull Put Spread":
        puts = opt_chain.puts
        otm_puts = puts[(puts['strike'] < current_price) & 
                       (puts['volume'] > 0) & 
                       (puts['openInterest'] > 10)].sort_values('strike', ascending=False)
        otm_puts = otm_puts.head(20)
        
        for i in range(len(otm_puts)):
            short_put = otm_puts.iloc[i]
            for j in range(i + 1, min(i + 10, len(otm_puts))):
                long_put = otm_puts.iloc[j]
                
                short_strike = short_put['strike']
                long_strike = long_put['strike']
                
                short_mid = (short_put['bid'] + short_put['ask']) / 2
                long_mid = (long_put['bid'] + long_put['ask']) / 2
                
                if short_mid <= long_mid:
                    continue
                
                net_credit = (short_mid - long_mid) * 100
                width = short_strike - long_strike
                max_risk = (width * 100) - net_credit
                
                if net_credit < 10 or max_risk <= 0:
                    continue
                
                short_iv = short_put['impliedVolatility'] if short_put['impliedVolatility'] > 0 else hist_volatility
                long_iv = long_put['impliedVolatility'] if long_put['impliedVolatility'] > 0 else hist_volatility
                
                spread_greeks = calculate_spread_greeks(
                    current_price, short_strike, long_strike, T_years, 
                    risk_free_rate, short_iv, long_iv, "Bull Put"
                )
                
                pop = calculate_probability_of_profit(
                    current_price, short_strike, T_years, risk_free_rate, 
                    short_iv, "Bull Put", net_credit, width
                )
                
                roi = (net_credit / max_risk) * 100
                ev = calculate_expected_value(net_credit, max_risk, pop)
                kelly = calculate_kelly_criterion(pop, net_credit, max_risk)
                daily_theta = spread_greeks['theta'] * 100
                score = calculate_score(roi, pop, ev, kelly, daily_theta)
                ai_rating = calculate_ai_rating(score, pop, ev, sentiment_val, spread_type)
                
                spreads.append({
                    'Short Strike': short_strike,
                    'Long Strike': long_strike,
                    'Width': width,
                    'Credit': net_credit,
                    'Max Risk': max_risk,
                    'ROI': roi,
                    'Prob. Profit': pop,
                    'Exp. Value': ev,
                    'Kelly %': kelly,
                    'AI Option': ai_rating,
                    'Θ/Day': daily_theta,
                    'Δ': spread_greeks['delta'] * 100,
                    'Γ': spread_greeks['gamma'] * 100,
                    'Score': score,
                    'short_iv': short_iv,
                    'long_iv': long_iv
                })
    
    else:  # Bear Call Spread
        calls = opt_chain.calls
        otm_calls = calls[(calls['strike'] > current_price) & 
                         (calls['volume'] > 0) & 
                         (calls['openInterest'] > 10)].sort_values('strike', ascending=True)
        otm_calls = otm_calls.head(20)
        
        for i in range(len(otm_calls)):
            short_call = otm_calls.iloc[i]
            for j in range(i + 1, min(i + 10, len(otm_calls))):
                long_call = otm_calls.iloc[j]
                
                short_strike = short_call['strike']
                long_strike = long_call['strike']
                
                short_mid = (short_call['bid'] + short_call['ask']) / 2
                long_mid = (long_call['bid'] + long_call['ask']) / 2
                
                if short_mid <= long_mid:
                    continue
                
                net_credit = (short_mid - long_mid) * 100
                width = long_strike - short_strike
                max_risk = (width * 100) - net_credit
                
                if net_credit < 10 or max_risk <= 0:
                    continue
                
                short_iv = short_call['impliedVolatility'] if short_call['impliedVolatility'] > 0 else hist_volatility
                long_iv = long_call['impliedVolatility'] if long_call['impliedVolatility'] > 0 else hist_volatility
                
                spread_greeks = calculate_spread_greeks(
                    current_price, short_strike, long_strike, T_years,
                    risk_free_rate, short_iv, long_iv, "Bear Call"
                )
                
                pop = calculate_probability_of_profit(
                    current_price, short_strike, T_years, risk_free_rate,
                    short_iv, "Bear Call", net_credit, width
                )
                
                roi = (net_credit / max_risk) * 100
                ev = calculate_expected_value(net_credit, max_risk, pop)
                kelly = calculate_kelly_criterion(pop, net_credit, max_risk)
                daily_theta = spread_greeks['theta'] * 100
                score = calculate_score(roi, pop, ev, kelly, daily_theta)
                ai_rating = calculate_ai_rating(score, pop, ev, sentiment_val, spread_type)
                
                spreads.append({
                    'Short Strike': short_strike,
                    'Long Strike': long_strike,
                    'Width': width,
                    'Credit': net_credit,
                    'Max Risk': max_risk,
                    'ROI': roi,
                    'Prob. Profit': pop,
                    'Exp. Value': ev,
                    'Kelly %': kelly,
                    'AI Option': ai_rating,
                    'Θ/Day': daily_theta,
                    'Δ': spread_greeks['delta'] * 100,
                    'Γ': spread_greeks['gamma'] * 100,
                    'Score': score,
                    'short_iv': short_iv,
                    'long_iv': long_iv
                })
    return spreads


@st.cache_data(ttl=60*60*24)
def get_constituent_stocks():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    nasdaq100_url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    
    try:
        sp500_table = pd.read_html(sp500_url)[0]
        sp500_symbols = sp500_table['Symbol'].tolist()
    except Exception as e:
        print(f"Could not fetch S&P 500 constituents: {e}")
        sp500_symbols = []

    try:
        # The correct table seems to be the 4th one on the page
        nasdaq100_table = pd.read_html(nasdaq100_url)[4]
        nasdaq100_symbols = nasdaq100_table['Ticker'].tolist()
    except Exception as e:
        print(f"Could not fetch Nasdaq 100 constituents: {e}")
        nasdaq100_symbols = []

    # Combine lists and remove duplicates
    combined_symbols = sorted(list(set(sp500_symbols + nasdaq100_symbols)))
    
    # Handle cases where scraping fails
    if not combined_symbols:
        # Fallback to a static list if scraping fails
        return ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'UNH', 'V', 'LLY', 'XOM']
        
    return combined_symbols

def calculate_health_score(financials_df):
    score = 0
    breakdown = []
    
    def add_item(name, points, max_points, desc):
        nonlocal score
        score += points
        breakdown.append({'name': name, 'score': points, 'max': max_points, 'desc': desc})

    try:
        # 1. Revenue Growth (YoY) - 30 points
        yoy_growth = (financials_df['Total Revenue'].iloc[0] / financials_df['Total Revenue'].iloc[-1]) - 1
        if yoy_growth > 0.10: 
            add_item('Revenue Growth', 30, 30, f">10% Growth ({yoy_growth:.1%})")
        elif yoy_growth > 0: 
            add_item('Revenue Growth', 15, 30, f"Positive Growth ({yoy_growth:.1%})")
        else:
            add_item('Revenue Growth', 0, 30, f"Negative Growth ({yoy_growth:.1%})")
    except (IndexError, ZeroDivisionError, KeyError):
        add_item('Revenue Growth', 0, 30, "Data Unavailable")

    try:
        # 2. Net Income Profitability - 30 points
        positive_quarters = (financials_df['Net Income'] > 0).sum()
        if positive_quarters == 4: 
            add_item('Profitability', 30, 30, "Profitable last 4 qtrs")
        elif positive_quarters >= 2: 
            add_item('Profitability', 15, 30, f"Profitable {positive_quarters}/4 qtrs")
        else:
            add_item('Profitability', 0, 30, f"Profitable {positive_quarters}/4 qtrs")
    except (IndexError, KeyError):
        add_item('Profitability', 0, 30, "Data Unavailable")

    try:
        # 3. Operating Cash Flow - 25 points
        if 'Operating Cash Flow' in financials_df.columns and (financials_df['Operating Cash Flow'] > 0).all():
            add_item('Cash Flow', 25, 25, "Positive OCF last 4 qtrs")
        elif 'Operating Cash Flow' in financials_df.columns and (financials_df['Operating Cash Flow'].iloc[:2] > 0).all():
            add_item('Cash Flow', 10, 25, "Positive OCF last 2 qtrs")
        else:
            add_item('Cash Flow', 0, 25, "Inconsistent/Negative OCF")
    except (IndexError, KeyError):
        add_item('Cash Flow', 0, 25, "Data Unavailable")

    try:
        # 4. Net Margin - 15 points
        avg_margin = (financials_df['Net Income'] / financials_df['Total Revenue']).mean()
        if avg_margin > 0.20: 
            add_item('Net Margin', 15, 15, f"Strong Margins ({avg_margin:.1%})")
        elif avg_margin > 0.10: 
            add_item('Net Margin', 10, 15, f"Healthy Margins ({avg_margin:.1%})")
        elif avg_margin > 0: 
            add_item('Net Margin', 5, 15, f"Positive Margins ({avg_margin:.1%})")
        else:
            add_item('Net Margin', 0, 15, f"Negative Margins ({avg_margin:.1%})")
    except (IndexError, ZeroDivisionError, KeyError):
        add_item('Net Margin', 0, 15, "Data Unavailable")
        
    return {'score': int(score), 'breakdown': breakdown}

@st.cache_data(ttl=60*60*4) # Cache for 4 hours
def generate_market_recommendations(_today_str): # The date string invalidates the cache daily
    all_symbols = get_constituent_stocks()
    import random
    # Replace symbols with '.' to '-' for yfinance compatibility
    all_symbols = [s.replace('.', '-') for s in all_symbols]
    stock_list = random.sample(all_symbols, min(20, len(all_symbols)))
    
    market_recommendations = []
    
    progress_bar = st.progress(0, text="Initializing market scan...")

    for i, symbol in enumerate(stock_list):
        progress_text = f"Scanning {symbol} ({i+1}/{len(stock_list)})..."
        progress_bar.progress((i) / len(stock_list), text=progress_text)
        
        try:
            stock_rec = yf.Ticker(symbol)
            hist_rec = stock_rec.history(period='1y')
            if hist_rec.empty:
                continue

            current_price_rec = hist_rec['Close'].iloc[-1]
            hist_vol_rec = hist_rec['Close'].pct_change().dropna().std() * np.sqrt(252)
            risk_free_rate_rec = 0.045
            
            all_spreads_for_stock = []
            expirations_rec = stock_rec.options
            valid_exps_rec = [e for e in expirations_rec if 7 < (datetime.strptime(e, '%Y-%m-%d') - datetime.now()).days < 90]

            for exp_date in valid_exps_rec[:5]: # Limit expirations to speed it up
                opt_chain_rec = stock_rec.option_chain(exp_date)
                days_to_exp_rec = (datetime.strptime(exp_date, '%Y-%m-%d') - datetime.now()).days
                T_years_rec = days_to_exp_rec / 365.0
                
                sentiment_val = get_news_sentiment(symbol)
                for stype in ["Bull Put Spread", "Bear Call Spread"]:
                    spreads_for_exp = scan_spread_opportunities(opt_chain_rec, current_price_rec, T_years_rec, hist_vol_rec, risk_free_rate_rec, stype, sentiment_val)
                    for spread in spreads_for_exp:
                        spread['Expiration'] = exp_date
                        spread['Type'] = stype
                        all_spreads_for_stock.append(spread)
            
            if all_spreads_for_stock:
                best_spread = max(all_spreads_for_stock, key=lambda x: x['Score'])
                best_spread['Symbol'] = symbol
                market_recommendations.append(best_spread)
        except Exception as e:
            # Silently fail for a single stock, so the whole process doesn't stop
            print(f"Could not scan {symbol}: {e}")
            continue

    progress_bar.progress(1.0, text="Scan complete!")
    
    if market_recommendations:
        rec_df = pd.DataFrame(market_recommendations).sort_values('Score', ascending=False)
        return rec_df.head(10)
    return pd.DataFrame()

@st.cache_data(ttl=60*60*4)
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl=60*60*24)
def get_constituent_stocks():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    nasdaq100_url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    
    try:
        sp500_table = pd.read_html(sp500_url)[0]
        sp500_symbols = sp500_table['Symbol'].tolist()
    except Exception as e:
        print(f"Could not fetch S&P 500 constituents: {e}")
        sp500_symbols = []

    try:
        # The correct table seems to be the 4th one on the page
        nasdaq100_table = pd.read_html(nasdaq100_url)[4]
        nasdaq100_symbols = nasdaq100_table['Ticker'].tolist()
    except Exception as e:
        print(f"Could not fetch Nasdaq 100 constituents: {e}")
        nasdaq100_symbols = []

    # Combine lists and remove duplicates
    combined_symbols = sorted(list(set(sp500_symbols + nasdaq100_symbols)))
    
    # Handle cases where scraping fails
    if not combined_symbols:
        # Fallback to a static list if scraping fails
        return ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'UNH', 'V', 'LLY', 'XOM']
        
    return combined_symbols

def create_stock_chart(hist, symbol):
    """Create stock price chart with daily bars (red/green) and RSI"""
    # Use last 90 days for the chart
    chart_data = hist.tail(90).copy()
    chart_data = chart_data.reset_index()
    chart_data['Date'] = pd.to_datetime(chart_data['Date']).dt.date
    
    # Calculate 30-day moving average
    chart_data['MA30'] = chart_data['Close'].rolling(window=30).mean()
    
    # Calculate RSI
    chart_data['RSI'] = calculate_rsi(chart_data['Close'], 14)
    
    # Determine if day was up or down
    chart_data['PrevClose'] = chart_data['Close'].shift(1)
    chart_data['Color'] = chart_data.apply(
        lambda x: 'green' if x['Close'] >= x['PrevClose'] else 'red', axis=1
    )
    chart_data.loc[chart_data.index[0], 'Color'] = 'green'  # First day default
    
    # Create bars for daily price movement (High to Low range)
    bars = alt.Chart(chart_data).mark_bar(size=8).encode(
        x=alt.X('Date:T', title='', axis=alt.Axis(format='%b %d', labelAngle=-45)),
        y=alt.Y('Low:Q', title='Price', scale=alt.Scale(zero=False)),
        y2='High:Q',
        color=alt.Color('Color:N', 
            scale=alt.Scale(domain=['green', 'red'], range=['#00ff7f', '#ff5555']),
            legend=None
        ),
        tooltip=[
            alt.Tooltip('Date:T', title='Date'),
            alt.Tooltip('Open:Q', title='Open', format='$.2f'),
            alt.Tooltip('High:Q', title='High', format='$.2f'),
            alt.Tooltip('Low:Q', title='Low', format='$.2f'),
            alt.Tooltip('Close:Q', title='Close', format='$.2f'),
            alt.Tooltip('MA30:Q', title='30-Day MA', format='$.2f')
        ]
    )
    
    # MA line overlay
    ma_line = alt.Chart(chart_data).mark_line(
        color='#f0c674',
        strokeWidth=2,
        strokeDash=[4, 2]
    ).encode(
        x='Date:T',
        y='MA30:Q'
    )
    
    # Combine price chart
    price_chart = (bars + ma_line).properties(
        height=280,
        title=f'{symbol} Daily Chart (90 Days) — Yellow: 30-Day MA'
    ).configure_axis(
        labelColor='#969896',
        titleColor='#969896',
        gridColor='#373b41',
        domainColor='#373b41'
    ).configure_view(
        strokeWidth=0
    ).configure_title(
        color='#ffffff',
        fontSize=14
    )
    
    # RSI Chart - separate
    rsi_line = alt.Chart(chart_data).mark_line(
        color='#b294bb',
        strokeWidth=2
    ).encode(
        x=alt.X('Date:T', title='', axis=alt.Axis(format='%b %d', labelAngle=-45)),
        y=alt.Y('RSI:Q', title='RSI', scale=alt.Scale(domain=[0, 100])),
        tooltip=[
            alt.Tooltip('Date:T', title='Date'),
            alt.Tooltip('RSI:Q', title='RSI', format='.1f')
        ]
    )
    
    # RSI area fill
    rsi_area = alt.Chart(chart_data).mark_area(
        color='#b294bb',
        opacity=0.2
    ).encode(
        x='Date:T',
        y='RSI:Q',
        y2=alt.datum(50)
    )
    
    # RSI overbought/oversold lines
    overbought = alt.Chart(pd.DataFrame({'y': [70]})).mark_rule(
        color='#ff5555',
        strokeDash=[4, 4],
        strokeWidth=1
    ).encode(y='y:Q')
    
    oversold = alt.Chart(pd.DataFrame({'y': [30]})).mark_rule(
        color='#00ff7f',
        strokeDash=[4, 4],
        strokeWidth=1
    ).encode(y='y:Q')
    
    middle = alt.Chart(pd.DataFrame({'y': [50]})).mark_rule(
        color='#5c6168',
        strokeDash=[2, 2],
        strokeWidth=1
    ).encode(y='y:Q')
    
    rsi_chart = (oversold + overbought + middle + rsi_area + rsi_line).properties(
        height=100,
        title='RSI (14)'
    ).configure_axis(
        labelColor='#969896',
        titleColor='#969896',
        gridColor='#373b41',
        domainColor='#373b41'
    ).configure_view(
        strokeWidth=0
    ).configure_title(
        color='#ffffff',
        fontSize=14
    )
    
    current_rsi = chart_data['RSI'].iloc[-1] if not pd.isna(chart_data['RSI'].iloc[-1]) else 50
    
    return price_chart, rsi_chart, current_rsi

def display_simulation_block(title, mc_results):
    st.markdown(f"""
    <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
        <div style="font-size: 1.1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 1rem;">
            {title}
        </div>
        """, unsafe_allow_html=True)
        
    r1, r2, r3 = st.columns(3)
    
    with r1:
        win_color = "metric-value-green" if mc_results['win_rate'] >= 60 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Simulated Win Rate</div>
            <div class="metric-value {win_color}">{mc_results['win_rate']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with r2:
        ev_color = "metric-value-green" if mc_results['expected_pnl'] > 0 else "metric-value-red"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean P&L</div>
            <div class="metric-value {ev_color}">${mc_results['expected_pnl']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        median_color = "metric-value-green" if mc_results['median_pnl'] > 0 else "metric-value-red"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Median P&L</div>
            <div class="metric-value {median_color}">${mc_results['median_pnl']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    r4, r5, r6 = st.columns(3)

    with r4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Std. Deviation</div>
            <div class="metric-value">${mc_results['std_dev']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with r5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Worst Case (5%)</div>
            <div class="metric-value metric-value-red">${mc_results['percentile_5']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with r6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Case (95%)</div>
            <div class="metric-value metric-value-green">${mc_results['percentile_95']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    pnl_df = pd.DataFrame({'pnl': mc_results['pnl_distribution']})

    base = alt.Chart(pnl_df).properties(
        height=300,
        title="P&L Distribution"
    )

    # Density plot for profit
    profit_density = base.transform_filter(
        alt.datum.pnl >= 0
    ).transform_density(
        'pnl',
        as_=['pnl_val', 'density'],
        bandwidth=max(round(mc_results['std_dev'] / 5), 5)
    ).mark_area(
        orient='vertical',
        color='#00ff7f',
        opacity=0.7,
        line={'color': '#00ff7f'}
    ).encode(
        x=alt.X('pnl_val:Q', title='Profit / Loss ($)'),
        y=alt.Y('density:Q', title='Probability Density'),
        tooltip=[
            alt.Tooltip('pnl_val:Q', title='P/L', format='$,.2f')
        ]
    )

    # Density plot for loss
    loss_density = base.transform_filter(
        alt.datum.pnl < 0
    ).transform_density(
        'pnl',
        as_=['pnl_val', 'density'],
        bandwidth=max(round(mc_results['std_dev'] / 5), 5)
    ).mark_area(
        orient='vertical',
        color='#ff5555',
        opacity=0.7,
        line={'color': '#ff5555'}
    ).encode(
        x=alt.X('pnl_val:Q', title='Profit / Loss ($)'),
        y=alt.Y('density:Q', title='Probability Density'),
        tooltip=[
            alt.Tooltip('pnl_val:Q', title='P/L', format='$,.2f')
        ]
    )

    # A line at P&L = 0
    zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
        color='white',
        strokeWidth=2,
        strokeDash=[3, 3]
    ).encode(x='x:Q')

    chart = (profit_density + loss_density + zero_line).configure_axis(
        labelColor='#969896',
        titleColor='#969896',
        gridColor='#373b41'
    ).configure_view(
        strokeWidth=0
    ).configure_title(
        color='#ffffff',
        fontSize=14,
        anchor='middle'
    )
    
    st.altair_chart(chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# 🖥️ MAIN APPLICATION UI
# ==========================================

# Hero Section
st.markdown("""
<div style="text-align: center; padding: 0.5rem 0 1rem 0;">
    <div class="hero-title">Options.AI</div>
    <div class="hero-subtitle">Futuristic-Grade Credit Spread Intelligence</div>
</div>
""", unsafe_allow_html=True)

# Main Tabs
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(["🌌 Knowledge-Base", "🚀 Opportunity-Scanner", "✨ Top-Picks", "⚡ Financial-Pulse"])

# ==========================================
# 🌌 KNOWLEDGE-BASE TAB (Now first)
# ==========================================
with main_tab1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="max-width: 900px; margin: 0 auto; padding: 1rem 0;">
    """, unsafe_allow_html=True)
    
    st.markdown("## 🎓 Understanding Credit Spreads")
    st.markdown("##### Everything explained in plain English")
    
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    
    # What is a Credit Spread
    with st.expander("What is a Credit Spread?", expanded=True):
        st.markdown("""
        **Think of it like getting paid to make a bet with limited downside.**
        
        A credit spread is a trading strategy where you:
        1. **Sell** an option (collect money upfront)
        2. **Buy** a cheaper option as insurance (pay a little back)
        3. **Keep the difference** if the stock behaves
        
        **Example:** 
        - Stock is at $100
        - You sell a put at $90 strike → collect $3
        - You buy a put at $85 strike → pay $1
        - **Net credit: $2** (this is yours to keep if stock stays above $90)
        
        **Why do this?**
        - You profit from stocks doing *nothing* or moving in your favor
        - Time is on your side (options lose value every day)
        - Your risk is capped (you know max loss upfront)
        """)
    
    # Bull Put vs Bear Call
    with st.expander("Bull Put Spread vs Bear Call Spread"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🐂 Bull Put Spread
            
            **Your bet:** Stock stays flat or goes UP
            
            **When to use:**
            - You're bullish (optimistic)
            - You think stock won't drop much
            - You want to profit from a stock NOT falling
            
            **How it works:**
            - Sell a put BELOW current price
            - Buy a cheaper put even further below
            - Collect the premium difference
            
            **You win if:** Stock stays above your short strike at expiration
            """)
        
        with col2:
            st.markdown("""
            ### 🐻 Bear Call Spread
            
            **Your bet:** Stock stays flat or goes DOWN
            
            **When to use:**
            - You're bearish (pessimistic)
            - You think stock won't rise much
            - You want to profit from a stock NOT rising
            
            **How it works:**
            - Sell a call ABOVE current price
            - Buy a cheaper call even further above
            - Collect the premium difference
            
            **You win if:** Stock stays below your short strike at expiration
            """)
    
    # Understanding the Scanner Table
    with st.expander("Understanding the Scanner Table"):
        st.markdown("""
        Here's what each column means:
        
        | Column | What it Means | Good Values |
        |--------|---------------|-------------|
        | **Short Strike** | The strike price you're SELLING (more risk, more reward) | Depends on your outlook |
        | **Long Strike** | The strike price you're BUYING (your insurance) | Further = more risk |
        | **Credit** | Money you receive upfront (per contract = 100 shares) | Higher is better |
        | **Max Risk** | Most you can lose if trade goes completely wrong | Lower is better |
        | **ROI** | Return on Investment = Credit ÷ Max Risk | Higher is better (15%+ is good) |
        | **Win %** | Probability the trade will be profitable | Higher is better (65%+ is good) |
        | **Exp. Value** | Average expected profit/loss per trade | Positive is good |
        | **Kelly %** | Suggested position size (% of your account) | 1-5% is conservative |
        | **Θ/Day** | Theta = money you make each day from time decay | Positive is good |
        | **Score** | Overall trade quality (combines all factors) | Higher is better (50+ is good) |
        """)
    
    # Understanding the Metrics
    with st.expander("Understanding the Market Overview"):
        st.markdown("""
        ### The Numbers at the Top
        
        **Current Price**
        - The stock's current trading price
        - The % shows today's change (green = up, red = down)
        
        **Historical Volatility (HV)**
        - How much the stock typically moves in a year
        - 30% HV = stock moves roughly 30% up or down per year
        - Higher volatility = bigger swings = more premium to collect
        
        **IV Rank**
        - Compares current volatility to the past year
        - **High IV Rank (>50%)** = Options are expensive → GOOD time to sell spreads
        - **Low IV Rank (<30%)** = Options are cheap → Maybe wait
        
        **5-Day / 30-Day Move**
        - Shows recent momentum
        - Helps you decide bullish vs bearish strategy
        """)
    
    # Understanding the Analysis
    with st.expander("Understanding the Trade Analysis"):
        st.markdown("""
        ### When You Click a Spread
        
        **Max Profit** 💰
        - The most you can make (the credit you receive)
        - You keep this if stock expires beyond your short strike
        
        **Max Risk** ⚠️
        - The most you can lose
        - = (Width between strikes × 100) - Credit received
        - This happens if stock goes completely against you
        
        **Breakeven** 📍
        - The exact price where you neither win nor lose
        - Bull Put: Short Strike - (Credit ÷ 100)
        - Bear Call: Short Strike + (Credit ÷ 100)
        
        **Risk/Reward Ratio**
        - How much you risk to make $1
        - 2:1 means you risk $2 to make $1
        - Lower is better, but usually means lower win rate
        """)
    
    # Understanding Greeks
    with st.expander("Understanding the Greeks"):
        st.markdown("""
        ### The Greeks (Don't worry, they're simpler than they sound!)
        
        **Delta (Δ)** - *Directional Risk*
        - How much your position changes when stock moves $1
        - Delta of 10 = you make/lose $10 per $1 stock move
        - **For credit spreads:** Lower delta = less directional risk
        
        **Gamma (Γ)** - *Delta's Rate of Change*
        - How fast your delta changes
        - **For credit spreads:** Lower gamma = more stable position
        
        **Theta (Θ)** - *Time Decay* ⭐ YOUR BEST FRIEND
        - Money you make each day just from time passing
        - **For credit spreads:** POSITIVE theta = you make money daily
        - This is why credit spreads work!
        
        **Vega** - *Volatility Sensitivity*
        - How much your position changes when volatility changes
        - **For credit spreads:** Negative vega = you profit when volatility drops
        """)
    
    # Understanding Monte Carlo
    with st.expander("Understanding Monte Carlo Simulation"):
        st.markdown("""
        ### What is Monte Carlo Simulation?
        
        **Imagine flipping a coin 10,000 times to see how often you'd win.**
        
        That's basically what Monte Carlo does, but for stock prices:
        
        1. We simulate the stock price moving randomly 10,000+ times
        2. Each simulation follows realistic market behavior
        3. We count how many times your trade would have been profitable
        4. This gives us a realistic win rate estimate
        
        **Why it matters:**
        - More accurate than simple probability formulas
        - Shows you the range of possible outcomes
        - Helps you understand the risk distribution
        
        **Reading the Results:**
        - **Simulated Win Rate:** % of simulations where you made money
        - **Expected P&L:** Average profit/loss across all simulations
        - **5th Percentile:** Worst case (95% of outcomes are better than this)
        - **95th Percentile:** Best case (only 5% of outcomes are better)
        """)
    
    # Quick Tips
    with st.expander("Quick Tips for Beginners"):
        st.markdown("""
        ### Do's ✅
        
        - **Start small** - Trade 1 contract until you're comfortable
        - **Look for high IV Rank** - Sell when options are expensive
        - **Aim for 65%+ win rate** - Stack the odds in your favor
        - **Use 30-45 day expirations** - Sweet spot for theta decay
        - **Set a max loss** - Exit if trade goes against you (e.g., at 2x credit)
        
        ### Don'ts ❌
        
        - **Don't risk more than 2-5% per trade** - Preserve your capital
        - **Don't hold through earnings** - Volatility can crush you
        - **Don't chase high ROI alone** - Usually means low win rate
        - **Don't ignore the trend** - Don't sell bull puts in a crash
        - **Don't forget to close winners** - Take profits at 50-75% of max
        
        ### The Golden Rule 🏆
        
        **Consistency > Home Runs**
        
        Credit spreads are about making small, consistent profits over time.
        A 70% win rate with 20% ROI beats a 30% win rate with 100% ROI.
        """)
    
    # Glossary
    with st.expander("Quick Glossary"):
        st.markdown("""
        | Term | Simple Definition |
        |------|-------------------|
        | **Strike Price** | The price at which the option can be exercised |
        | **Premium** | The price/cost of an option |
        | **Expiration** | The date when the option expires |
        | **OTM (Out of the Money)** | Option that has no current value to exercise |
        | **ITM (In the Money)** | Option that has value if exercised now |
        | **ATM (At the Money)** | Strike price equals current stock price |
        | **Credit** | Money received when opening a trade |
        | **Debit** | Money paid when opening a trade |
        | **Width** | Difference between your two strike prices |
        | **Assignment** | When you're forced to buy/sell shares |
        | **Theta Decay** | Options losing value as time passes |
        | **IV Crush** | Volatility dropping (usually after events) |
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-top: 2rem; background: linear-gradient(135deg, #16161f 0%, #12121a 100%); border-radius: 16px; border: 1px solid #2a2a3a;">
        <div style="font-size: 1.25rem; color: #f0f0f5; margin-bottom: 0.5rem;">Ready to start scanning?</div>
        <div style="color: #8888a0; font-size: 0.9rem;">Head over to the Scanner tab and find your first trade!</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 🚀 OPPORTUNITY-SCANNER TAB (Now second)
# ==========================================
with main_tab2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    # Input Section
    st.markdown('<div class="section-header">Configure Scanner</div>', unsafe_allow_html=True)

    col_input1, col_input2, col_input3 = st.columns([2, 3, 1])

    with col_input1:
        symbol = st.text_input(
            "Symbol",
            value="NVDA",
            placeholder="Enter ticker...",
            key="symbol_input",
            label_visibility="collapsed"
        ).upper()

    with col_input2:
        spread_type = st.radio(
            "Strategy",
            ["Bull Put Spread", "Bear Call Spread"],
            horizontal=True,
            key="spread_type",
            label_visibility="collapsed"
        )

    with col_input3:
        scan_clicked = st.button("Scan", use_container_width=True, type="primary")

    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'selected_spread' not in st.session_state:
        st.session_state.selected_spread = None

    # Main Data Loading
    if symbol and (scan_clicked or st.session_state.data_loaded):
        try:
            with st.spinner(f'Analyzing {symbol} options chain...'):
                stock = yf.Ticker(symbol)
                
                hist = stock.history(period='1y')
                
                if hist.empty:
                    st.error(f"❌ Could not fetch data for {symbol}. Please check the symbol and try again.")
                    st.stop()
                
                current_price = hist['Close'].iloc[-1]
                daily_returns = hist['Close'].pct_change().dropna()
                hist_volatility = daily_returns.std() * np.sqrt(252)
                
                rolling_vol = daily_returns.rolling(window=20).std() * np.sqrt(252)
                rolling_vol = rolling_vol.dropna()
                
                price_change_1d = ((current_price / hist['Close'].iloc[-2]) - 1) * 100 if len(hist) > 1 else 0
                price_change_5d = ((current_price / hist['Close'].iloc[-5]) - 1) * 100 if len(hist) > 5 else 0
                price_change_30d = ((current_price / hist['Close'].iloc[-22]) - 1) * 100 if len(hist) > 22 else 0
                
                st.session_state.data_loaded = True
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            # Stock Chart Section
            st.markdown('<div class="section-header">Price Chart</div>', unsafe_allow_html=True)
            
            # Get both charts
            price_chart, rsi_chart, current_rsi = create_stock_chart(hist, symbol)
            
            # Display charts separately - each will fill full width
            st.altair_chart(price_chart, use_container_width=True)
            st.altair_chart(rsi_chart, use_container_width=True)
            
            # Signal Cards Row - All 3 in a row
            sig_col1, sig_col2, sig_col3 = st.columns(3)
            
            with sig_col1:
                if current_rsi > 70:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(255, 71, 87, 0.2) 0%, rgba(255, 71, 87, 0.1) 100%); 
                                border: 1px solid rgba(255, 71, 87, 0.4); border-radius: 10px; padding: 1rem; text-align: center; height: 100%;">
                        <div style="color: #ff4757; font-weight: 600;">🔴 OVERBOUGHT</div>
                        <div style="color: #ff4757; font-size: 1.5rem; font-weight: 700;">RSI: {current_rsi:.1f}</div>
                        <div style="color: #ff6b7a; font-size: 0.75rem;">Consider Bear Call Spread</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif current_rsi < 30:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(0, 212, 170, 0.2) 0%, rgba(0, 212, 170, 0.1) 100%); 
                                border: 1px solid rgba(0, 212, 170, 0.4); border-radius: 10px; padding: 1rem; text-align: center; height: 100%;">
                        <div style="color: #00d4aa; font-weight: 600;">🟢 OVERSOLD</div>
                        <div style="color: #00d4aa; font-size: 1.5rem; font-weight: 700;">RSI: {current_rsi:.1f}</div>
                        <div style="color: #00e6b8; font-size: 0.75rem;">Consider Bull Put Spread</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 1rem; text-align: center; height: 100%;">
                        <div style="color: #4a9eff; font-weight: 600;">⚖️ NEUTRAL</div>
                        <div style="color: #4a9eff; font-size: 1.5rem; font-weight: 700;">RSI: {current_rsi:.1f}</div>
                        <div style="color: var(--text-muted); font-size: 0.75rem;">Use trend for direction</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with sig_col2:
                ma_value = hist['Close'].rolling(30).mean().iloc[-1]
                above_ma = hist['Close'].iloc[-1] > ma_value
                if above_ma:
                    st.markdown(f"""
                    <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 1rem; text-align: center; height: 100%;">
                        <div style="color: #00d4aa; font-weight: 600;">✅ Above 30-Day MA</div>
                        <div style="color: var(--text-primary); font-size: 1.5rem; font-weight: 700;">${hist['Close'].iloc[-1]:.2f}</div>
                        <div style="color: var(--text-muted); font-size: 0.75rem;">MA: ${ma_value:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 1rem; text-align: center; height: 100%;">
                        <div style="color: #ff4757; font-weight: 600;">⚠️ Below 30-Day MA</div>
                        <div style="color: var(--text-primary); font-size: 1.5rem; font-weight: 700;">${hist['Close'].iloc[-1]:.2f}</div>
                        <div style="color: var(--text-muted); font-size: 0.75rem;">MA: ${ma_value:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with sig_col3:
                price_30d_ago = hist['Close'].iloc[-30] if len(hist) >= 30 else hist['Close'].iloc[0]
                trend_pct = ((hist['Close'].iloc[-1] / price_30d_ago) - 1) * 100
                if trend_pct > 0:
                    st.markdown(f"""
                    <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 1rem; text-align: center; height: 100%;">
                        <div style="color: #00d4aa; font-weight: 600;">📈 Bullish Trend</div>
                        <div style="color: #00d4aa; font-size: 1.5rem; font-weight: 700;">+{trend_pct:.1f}%</div>
                        <div style="color: var(--text-muted); font-size: 0.75rem;">30-day change</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 1rem; text-align: center; height: 100%;">
                        <div style="color: #ff4757; font-weight: 600;">📉 Bearish Trend</div>
                        <div style="color: #ff4757; font-size: 1.5rem; font-weight: 700;">{trend_pct:.1f}%</div>
                        <div style="color: var(--text-muted); font-size: 0.75rem;">30-day change</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
            
            # Stock Overview Section
            st.markdown('<div class="section-header">Market Overview</div>', unsafe_allow_html=True)
            
            col_price, m1, m2, m3, m4 = st.columns(5)
            
            with col_price:
                change_color = "metric-value-green" if price_change_1d >= 0 else "metric-value-red"
                change_sign = "+" if price_change_1d >= 0 else ""
                
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
                            border: 1px solid var(--border-color);
                            border-radius: 16px;
                            padding: 1.25rem;
                            min-height: 90px;
                            position: relative;
                            overflow: hidden;">
                    <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; 
                                background: linear-gradient(90deg, #00d4aa, #4a9eff, #9d4edd);"></div>
                    <div class="metric-label">{symbol}</div>
                    <div class="metric-value">${current_price:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Historical Vol</div>
                    <div class="metric-value">{hist_volatility*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m2:
                iv_rank = calculate_iv_rank(hist_volatility, rolling_vol.values)
                iv_color = "metric-value-green" if iv_rank > 50 else ""
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">IV Rank</div>
                    <div class="metric-value {iv_color}">{iv_rank:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">5-Day Move</div>
                    <div class="metric-value {"metric-value-green" if price_change_5d >= 0 else "metric-value-red"}">
                        {"+" if price_change_5d >= 0 else ""}{price_change_5d:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">30-Day Move</div>
                    <div class="metric-value {"metric-value-green" if price_change_30d >= 0 else "metric-value-red"}">
                        {"+" if price_change_30d >= 0 else ""}{price_change_30d:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Expiration Selection
            st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
            
            expirations = stock.options
                
            if not expirations:
                st.warning("⚠️ No options data available for this symbol.")
                st.stop()
            
            valid_exps = []
            for e in expirations:
                try:
                    days_to_exp = (datetime.strptime(e, '%Y-%m-%d') - datetime.now()).days
                    if 7 < days_to_exp < 90:
                        valid_exps.append(e)
                except ValueError:
                    continue
            
            if not valid_exps:
                valid_exps = expirations[:10]
            
            # Store expiration options for later use
            exp_options = valid_exps[:8]
            
            # Use first expiration as default for initial load
            default_exp = exp_options[0]
            days_to_exp_default = (datetime.strptime(default_exp, '%Y-%m-%d') - datetime.now()).days
            T_years_default = days_to_exp_default / 365.0
            risk_free_rate = 0.045
            
            # Display Results section with expiration selection
            st.markdown('<div class="section-header">Opportunity Scanner</div>', unsafe_allow_html=True)
            
            # Expiration Date Selection - styled like tabs
            st.markdown("""
            <div style="margin-bottom: 0.5rem;">
                <span style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted); 
                            text-transform: uppercase; letter-spacing: 0.1em;">Expiration Date</span>
            </div>
            """, unsafe_allow_html=True)
            
            selected_exp = st.radio(
                "Select Expiration",
                exp_options,
                horizontal=True,
                format_func=lambda x: f"{x} ({(datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days}d)",
                key="exp_radio",
                label_visibility="collapsed"
            )
            
            days_to_exp = (datetime.strptime(selected_exp, '%Y-%m-%d') - datetime.now()).days
            T_years = days_to_exp / 365.0
            
            opt_chain = stock.option_chain(selected_exp)
            
            # Get AI Sentiment
            sentiment_val = get_news_sentiment(symbol)
            spreads = scan_spread_opportunities(opt_chain, current_price, T_years, hist_volatility, risk_free_rate, spread_type, sentiment_val)
            
            # Build dataframe and show results
            if spreads:
                df = pd.DataFrame(spreads).sort_values('Score', ascending=False).head(50)
                
                st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
                
                top_spread = df.iloc[0]
                avg_pop = df['Prob. Profit'].mean()
                avg_ev = df['Exp. Value'].mean()
                
                stat1, stat2, stat3, stat4 = st.columns(4)
                with stat1:
                    st.metric("Spreads Found", len(df))
                with stat2:
                    st.metric("Top Score", f"{top_spread['Score']:.1f}")
                with stat3:
                    st.metric("Avg Win Rate", f"{avg_pop:.1f}%")
                with stat4:
                    st.metric("Avg Exp. Value", f"${avg_ev:.2f}")
                
                # Instruction to select
                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(74, 158, 255, 0.1) 0%, rgba(74, 158, 255, 0.05) 100%); 
                            border: 1px solid rgba(74, 158, 255, 0.3); border-radius: 10px; padding: 0.75rem 1rem; margin: 1rem 0;">
                    <span style="color: #4a9eff; font-weight: 600;">↓ Select a trade</span>
                    <span style="color: var(--text-secondary);"> to see P&L profile, Monte Carlo simulation, and Greeks analysis below</span>
                </div>
                """, unsafe_allow_html=True)
                
                display_df = df[['Short Strike', 'Long Strike', 'Credit', 'Max Risk', 'ROI', 
                               'Prob. Profit', 'Exp. Value', 'AI Option', 'Θ/Day', 'Score']].copy()
                
                def color_ai_option(val):
                    if val == 'Great':
                        return 'color: #00ff7f; font-weight: bold'
                    elif val == 'Good':
                        return 'color: #81a2be; font-weight: bold'
                    elif val == 'Fair':
                        return 'color: #f0c674'
                    else:
                        return 'color: #ff5555'
                
                selection = st.dataframe(
                    display_df.style.format({
                        'Short Strike': '${:.2f}',
                        'Long Strike': '${:.2f}',
                        'Credit': '${:.2f}',
                        'Max Risk': '${:.2f}',
                        'ROI': '{:.1f}%',
                        'Prob. Profit': '{:.1f}%',
                        'Exp. Value': '${:.2f}',
                        'Θ/Day': '${:.2f}',
                        'Score': '{:.1f}'
                    }).map(color_ai_option, subset=['AI Option']),
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    height=400,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score",
                            format="%.1f",
                            min_value=0,
                            max_value=100,
                        ),
                        "Prob. Profit": st.column_config.ProgressColumn(
                            "Win %",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "ROI": st.column_config.NumberColumn(
                            "ROI",
                            format="%.1f%%",
                        ),
                        "Exp. Value": st.column_config.NumberColumn(
                            "Exp. Value",
                            format="$%.2f",
                        ),
                    }
                )
                
                if selection.selection.rows:
                    selected_idx = selection.selection.rows[0]
                    selected_row = df.iloc[selected_idx]
                    st.session_state.selected_spread = selected_row
                
                # Analysis Panel
                if st.session_state.selected_spread is not None:
                    sel = st.session_state.selected_spread
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    short_k = sel['Short Strike']
                    long_k = sel['Long Strike']
                    credit = sel['Credit']
                    calc_type = "Bull Put" if "Put" in spread_type else "Bear Call"
                    
                    st.markdown(f"""
                    <div class="section-header">Trade Analysis: {symbol} {calc_type} ${short_k}/{long_k}</div>
                    """, unsafe_allow_html=True)
                    
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📈 P&L Profile", 
                        "📊 Greeks & Analytics", 
                        "🎲 Monte Carlo", 
                        "🔥 High Volatility", 
                        "🧊 Low Volatility"
                    ])
                    
                    with tab1:
                        # Contract quantity selector
                        col_qty, col_empty1, col_empty2 = st.columns([1, 2, 2])
                        with col_qty:
                            quantity = st.number_input("Contracts", min_value=1, max_value=100, value=1, key="qty_pnl")
                        
                        total_credit = credit * quantity
                        total_risk = sel['Max Risk'] * quantity
                        breakeven = short_k - (credit / 100) if calc_type == "Bull Put" else short_k + (credit / 100)
                        
                        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
                        
                        # Key metrics
                        k1, k2, k3, k4, k5 = st.columns(5)
                        
                        with k1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Max Profit</div>
                                <div class="metric-value metric-value-green">${total_credit:,.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with k2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Max Risk</div>
                                <div class="metric-value metric-value-red">${total_risk:,.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with k3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Breakeven</div>
                                <div class="metric-value">${breakeven:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with k4:
                            rr_ratio = total_risk / total_credit
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Risk/Reward</div>
                                <div class="metric-value">{rr_ratio:.1f}:1</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with k5:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Trade Score</div>
                                <div class="metric-value">{sel['Score']:.0f}/100</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
                        
                        chart, be, ml = generate_pnl_chart(
                            short_k, long_k, total_credit, quantity, calc_type, current_price
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        leg1, leg2, leg3, leg4 = st.columns(4)
                        with leg1:
                            st.markdown("🔵 **Current Price**")
                        with leg2:
                            st.markdown("🟡 **Breakeven**")
                        with leg3:
                            st.markdown("🟢 **Profit Zone**")
                        with leg4:
                            st.markdown("🔴 **Loss Zone**")
                    
                    with tab2:
                        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
                        
                        st.markdown("**Position Greeks** (per contract)")
                        
                        g1, g2, g3, g4 = st.columns(4)
                        
                        with g1:
                            st.markdown(f"""
                            <div class="greek-item">
                                <div class="greek-label">Delta (Δ)</div>
                                <div class="greek-value">{sel['Δ']:+.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.caption("Directional exposure")
                        
                        with g2:
                            st.markdown(f"""
                            <div class="greek-item">
                                <div class="greek-label">Gamma (Γ)</div>
                                <div class="greek-value">{sel['Γ']:+.3f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.caption("Delta sensitivity")
                        
                        with g3:
                            theta_color = "metric-value-green" if sel['Θ/Day'] > 0 else ""
                            st.markdown(f"""
                            <div class="greek-item">
                                <div class="greek-label">Theta (Θ)</div>
                                <div class="greek-value {theta_color}">${sel['Θ/Day']:+.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.caption("Daily time decay")
                        
                        with g4:
                            st.markdown(f"""
                            <div class="greek-item">
                                <div class="greek-label">Implied Vol</div>
                                <div class="greek-value">{sel['short_iv']*100:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.caption("Short strike IV")
                        
                        st.markdown("<hr>", unsafe_allow_html=True)
                        
                        st.markdown("**Risk Analytics**")
                        
                        a1, a2, a3 = st.columns(3)
                        
                        with a1:
                            st.metric("Expected Value", f"${sel['Exp. Value']:.2f}")
                            ev_pct = min(max((sel['Exp. Value'] + 50) / 100 * 100, 0), 100)
                            st.progress(ev_pct / 100)
                        
                        with a2:
                            st.metric("Kelly Criterion", f"{sel['Kelly %']:.1f}%")
                            kelly_pct = min(sel['Kelly %'] / 25 * 100, 100)
                            st.progress(kelly_pct / 100)
                        
                        with a3:
                            st.metric("Win Probability", f"{sel['Prob. Profit']:.1f}%")
                            st.progress(sel['Prob. Profit'] / 100)

                    avg_iv = (sel['short_iv'] + sel['long_iv']) / 2

                    with tab3: # Base Monte Carlo
                        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
                        mc_col1, mc_col2, mc_col3 = st.columns([1, 1, 2])
                        with mc_col1:
                            quantity_mc = st.number_input("Contracts", min_value=1, max_value=100, value=1, key="qty_mc_base")
                        with mc_col2:
                            n_sims = st.selectbox("Simulations", [5000, 10000, 25000], index=1, key="n_sims_base")
                        
                        total_credit_mc = credit * quantity_mc
                        
                        with st.spinner('Running base simulation...'):
                            mc_results_base = evaluate_spread_monte_carlo(
                                current_price, short_k, long_k, T_years, risk_free_rate,
                                avg_iv, calc_type, total_credit_mc, quantity_mc, n_sims
                            )
                        display_simulation_block(f"Base Scenario: Implied Volatility ({avg_iv*100:.1f}%)", mc_results_base)

                    with tab4: # High Volatility
                        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
                        mc_col1, mc_col2, mc_col3 = st.columns([1, 1, 2])
                        with mc_col1:
                            quantity_mc = st.number_input("Contracts", min_value=1, max_value=100, value=1, key="qty_mc_high")
                        with mc_col2:
                            n_sims = st.selectbox("Simulations", [5000, 10000, 25000], index=1, key="n_sims_high")

                        total_credit_mc = credit * quantity_mc
                        vol_multiplier = 1.25
                        
                        with st.spinner('Running high volatility simulation...'):
                            mc_results_high = evaluate_spread_monte_carlo(
                                current_price, short_k, long_k, T_years, risk_free_rate,
                                avg_iv * vol_multiplier, calc_type, total_credit_mc, quantity_mc, n_sims
                            )
                        display_simulation_block(f"Scenario: High Volatility ({avg_iv*vol_multiplier*100:.1f}%)", mc_results_high)

                    with tab5: # Low Volatility
                        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
                        mc_col1, mc_col2, mc_col3 = st.columns([1, 1, 2])
                        with mc_col1:
                            quantity_mc = st.number_input("Contracts", min_value=1, max_value=100, value=1, key="qty_mc_low")
                        with mc_col2:
                            n_sims = st.selectbox("Simulations", [5000, 10000, 25000], index=1, key="n_sims_low")

                        total_credit_mc = credit * quantity_mc
                        vol_multiplier = 0.75
                        
                        with st.spinner('Running low volatility simulation...'):
                            mc_results_low = evaluate_spread_monte_carlo(
                                current_price, short_k, long_k, T_years, risk_free_rate,
                                avg_iv * vol_multiplier, calc_type, total_credit_mc, quantity_mc, n_sims
                            )
                        display_simulation_block(f"Scenario: Low Volatility ({avg_iv*vol_multiplier*100:.1f}%)", mc_results_low)

            
            else:
                st.warning("⚠️ No viable spreads found. Try a different expiration date or spread type.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check the symbol and try again.")

    else:
        # Welcome state
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #16161f 0%, #12121a 100%); border-radius: 20px; border: 1px solid #2a2a3a; margin-top: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">◈</div>
            <div style="font-size: 1.5rem; color: #f0f0f5; font-weight: 600; margin-bottom: 0.5rem;">
                Enter a Symbol to Begin
            </div>
            <div style="color: #8888a0; font-size: 1rem; max-width: 500px; margin: 0 auto;">
                Options.AI scans the entire options chain to find optimal credit spread opportunities 
                using institutional-grade analytics and Monte Carlo simulation.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
        
        f1, f2, f3, f4 = st.columns(4)
        
        with f1:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-weight: 600; color: #f0f0f5; margin-bottom: 0.25rem;">Full Chain Scan</div>
                <div style="font-size: 0.8rem; color: #8888a0;">Analyzes every viable spread</div>
            </div>
            """, unsafe_allow_html=True)
        
        with f2:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎲</div>
                <div style="font-weight: 600; color: #f0f0f5; margin-bottom: 0.25rem;">Monte Carlo</div>
                <div style="font-size: 0.8rem; color: #8888a0;">10,000+ simulations</div>
            </div>
            """, unsafe_allow_html=True)
        
        with f3:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📈</div>
                <div style="font-weight: 600; color: #f0f0f5; margin-bottom: 0.25rem;">Greeks Analysis</div>
                <div style="font-size: 0.8rem; color: #8888a0;">Δ, Γ, Θ, Vega calculated</div>
            </div>
            """, unsafe_allow_html=True)
        
        with f4:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="font-weight: 600; color: #f0f0f5; margin-bottom: 0.25rem;">Kelly Sizing</div>
                <div style="font-size: 0.8rem; color: #8888a0;">Optimal position sizing</div>
            </div>
            """, unsafe_allow_html=True)

with main_tab3:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">✨ Top-Picks</div>', unsafe_allow_html=True)
    
    rec_col1, rec_col2 = st.columns([1, 3])
    with rec_col1:
        rec_symbol = st.text_input("Symbol for Recommendation", value="NVDA", key="rec_symbol_input").upper()

    with rec_col2:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        find_rec_clicked = st.button("🚀 Find Top 5 Trades", key="find_trades")

    if find_rec_clicked:
        with st.spinner(f"Scanning all expirations for {rec_symbol} to find the best trades... This may take a moment."):
            try:
                stock_rec = yf.Ticker(rec_symbol)
                hist_rec = stock_rec.history(period='1y')
                if hist_rec.empty:
                    st.error(f"Could not fetch data for {rec_symbol}.")
                    st.stop()

                current_price_rec = hist_rec['Close'].iloc[-1]
                hist_vol_rec = hist_rec['Close'].pct_change().dropna().std() * np.sqrt(252)
                risk_free_rate_rec = 0.045
                
                all_spreads = []
                expirations_rec = stock_rec.options
                valid_exps_rec = [e for e in expirations_rec if 7 < (datetime.strptime(e, '%Y-%m-%d') - datetime.now()).days < 90]

                for i, exp_date in enumerate(valid_exps_rec):
                    opt_chain_rec = stock_rec.option_chain(exp_date)
                    days_to_exp_rec = (datetime.strptime(exp_date, '%Y-%m-%d') - datetime.now()).days
                    T_years_rec = days_to_exp_rec / 365.0
                    
                    sentiment_val = get_news_sentiment(rec_symbol)
                    for stype in ["Bull Put Spread", "Bear Call Spread"]:
                        spreads_for_exp = scan_spread_opportunities(opt_chain_rec, current_price_rec, T_years_rec, hist_vol_rec, risk_free_rate_rec, stype, sentiment_val)
                        for spread in spreads_for_exp:
                            spread['Expiration'] = exp_date
                            spread['Type'] = stype
                            all_spreads.append(spread)
                
                if all_spreads:
                    rec_df = pd.DataFrame(all_spreads).sort_values('Score', ascending=False).head(5)
                    st.markdown("### Top 5 Recommended Trades")
                    
                    display_rec_df = rec_df[['Expiration', 'Type', 'Short Strike', 'Long Strike', 'Credit', 'Max Risk', 'ROI', 'Prob. Profit', 'Score']].copy()
                    
                    st.dataframe(
                        display_rec_df.style.format({
                            'Short Strike': '${:.2f}', 'Long Strike': '${:.2f}', 'Credit': '${:.2f}', 'Max Risk': '${:.2f}',
                            'ROI': '{:.1f}%', 'Prob. Profit': '{:.1f}%', 'Score': '{:.1f}'
                        }),
                        use_container_width=True, hide_index=True,
                        column_config={
                            "Score": st.column_config.ProgressColumn("Score", format="%.1f", min_value=0, max_value=100),
                            "Prob. Profit": st.column_config.ProgressColumn("Win %", format="%.1f%%", min_value=0, max_value=100),
                        }
                    )
                else:
                    st.warning("Could not find any viable trades across all expirations.")
            except Exception as e:
                st.error(f"An error occurred during recommendation scan: {e}")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Market-Wide Top 10 Opportunities</div>', unsafe_allow_html=True)
    
    st.info("""
    **Note:** This is a computationally intensive scan across multiple major stocks and their options chains. 
    The results are cached for 4 hours to provide a daily snapshot of the market.
    """)

    if st.button("📈 Generate Daily Market Report"):
        today_str = datetime.now().strftime("%Y-%m-%d")
        top_10_df = generate_market_recommendations(today_str)

        if not top_10_df.empty:
            st.markdown("### Top 10 Stocks by Best Trade Score")
            
            display_df = top_10_df[['Symbol', 'Score', 'Expiration', 'Type', 'Short Strike', 'Long Strike', 'ROI', 'Prob. Profit']].copy()
            display_df.reset_index(drop=True, inplace=True)
            display_df.index = display_df.index + 1
            
            st.dataframe(
                display_df.style.format({
                    'Short Strike': '${:.2f}', 'Long Strike': '${:.2f}',
                    'ROI': '{:.1f}%', 'Prob. Profit': '{:.1f}%', 'Score': '{:.1f}'
                }),
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Best Score", format="%.1f", min_value=0, max_value=100),
                    "Prob. Profit": st.column_config.ProgressColumn("Win %", format="%.1f%%", min_value=0, max_value=100),
                }
            )
        else:
            st.warning("Could not generate market recommendations. Please try again later.")
    st.markdown('</div>', unsafe_allow_html=True)

with main_tab4:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚡ Financial-Pulse</div>', unsafe_allow_html=True)
    fin_col1, fin_col2 = st.columns([1, 3])
    with fin_col1:
        fin_symbol = st.text_input("Symbol", value="NVDA", key="financials_symbol").upper()
    with fin_col2:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        get_financials_clicked = st.button("📊 Get Financial Data", key="get_financials")

    if get_financials_clicked:
        try:
            with st.spinner(f"Fetching financial statements for {fin_symbol}..."):
                stock_fin = yf.Ticker(fin_symbol)
                
                # Fetching quarterly data
                income_stmt_q = stock_fin.quarterly_income_stmt
                cash_flow_q = stock_fin.quarterly_cashflow
                
                if income_stmt_q.empty:
                    st.error(f"Could not fetch quarterly financial data for {fin_symbol}. The symbol may be incorrect or data may be unavailable.")
                    st.stop()

            st.markdown(f"### Financial Snapshot: {fin_symbol}")

            # Combine key data into one DataFrame
            key_financials = pd.DataFrame(index=["Total Revenue", "Gross Profit", "Operating Income", "EBITDA", "Net Income", "Operating Cash Flow"])
            
            # Find common columns (dates) and take the most recent 4
            common_dates = income_stmt_q.columns.intersection(cash_flow_q.columns)
            if len(common_dates) == 0:
                st.warning("Could not align financial statement dates. Data may be incomplete.")
                # Fallback to just income statement dates
                common_dates = income_stmt_q.columns

            for col in common_dates[:4]: 
                # Check if col exists in cash_flow_q just in case of fallback
                op_cash_flow = 'N/A'
                if col in cash_flow_q.columns and 'Operating Cash Flow' in cash_flow_q.index:
                    op_cash_flow = cash_flow_q.loc['Operating Cash Flow', col]

                key_financials[col.strftime('%Y-%m-%d')] = [
                    income_stmt_q.loc['Total Revenue', col],
                    income_stmt_q.loc['Gross Profit', col] if 'Gross Profit' in income_stmt_q.index else 'N/A',
                    income_stmt_q.loc['Operating Income', col] if 'Operating Income' in income_stmt_q.index else 'N/A',
                    income_stmt_q.loc['EBITDA', col] if 'EBITDA' in income_stmt_q.index else 'N/A',
                    income_stmt_q.loc['Net Income', col],
                    op_cash_flow
                ]
            
            # Calculate margins
            try:
                gross_margin = (key_financials.loc['Gross Profit'] / key_financials.loc['Total Revenue']) * 100
                net_margin = (key_financials.loc['Net Income'] / key_financials.loc['Total Revenue']) * 100
                key_financials.loc['Gross Margin (%)'] = gross_margin.to_list()
                key_financials.loc['Net Margin (%)'] = net_margin.to_list()
            except (TypeError, ZeroDivisionError): # Handles 'N/A' strings or zero revenue
                pass

            # Display formatted table
            st.dataframe(
                key_financials.style.format(
                    formatter="{:,.0f}", na_rep='N/A'
                ).format(
                    formatter="{:.1f}%", subset=(['Gross Margin (%)', 'Net Margin (%)'], slice(None))
                ),
                use_container_width=True
            )

            # Health Score
            st.markdown("<hr>", unsafe_allow_html=True)
            
            health_data = calculate_health_score(key_financials.T)
            score = health_data['score']
            
            score_color = "#00d4aa" if score >= 70 else "#ffd700" if score >= 40 else "#ff4757"
            
            breakdown_html = ""
            for item in health_data['breakdown']:
                bar_width = (item['score'] / item['max']) * 100
                item_color = "#00d4aa" if item['score'] == item['max'] else "#4a9eff" if item['score'] > 0 else "#5a5a70"
                
                breakdown_html += f"""<div style="background: var(--bg-elevated); border-radius: 12px; padding: 1rem; border: 1px solid var(--border-color);">
    <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">{item['name']}</div>
    <div style="display: flex; justify-content: space-between; align-items: end; margin-bottom: 0.5rem;">
        <div style="font-size: 1.1rem; font-weight: 600; color: var(--text-primary);">{item['score']} <span style="font-size: 0.8rem; color: var(--text-muted);">/ {item['max']}</span></div>
    </div>
    <div style="width: 100%; height: 4px; background: #2a2a3a; border-radius: 2px; margin-bottom: 0.5rem;">
        <div style="width: {bar_width}%; height: 100%; background: {item_color}; border-radius: 2px;"></div>
    </div>
    <div style="font-size: 0.75rem; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{item['desc']}</div>
</div>"""
            
            st.markdown(f"""<div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;">
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;">
        <div>
            <div style="font-size: 1.1rem; font-weight: 600; color: var(--text-primary);">Company Health Score</div>
            <div style="font-size: 0.8rem; color: var(--text-secondary);">Fundamental strength analysis based on last 4 quarters</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 2rem; font-weight: 700; color: {score_color};">{score} <span style="font-size: 1rem; color: var(--text-muted);">/ 100</span></div>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
        {breakdown_html}
    </div>
</div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Please check the ticker symbol. Some symbols (e.g., ETFs like SPY) do not have financial statements.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0; margin-top: 3rem; border-top: 1px solid #2a2a3a;">
    <div style="color: #5a5a70; font-size: 0.75rem;">
        ◈ Options.AI | For educational purposes only. Not financial advice.
    </div>
</div>
""", unsafe_allow_html=True)