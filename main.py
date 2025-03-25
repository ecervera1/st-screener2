import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from datetime import datetime, timedelta
import re
from prophet import Prophet
import numpy as np
import requests
from bs4 import BeautifulSoup
import seaborn as sns
from scipy.optimize import minimize
from pyfinviz.quote import Quote
import asyncio
from aiohttp_retry import RetryClient, ExponentialRetry
import logging

# ======================
# Configuration & Styling
# ======================
def configure_page():
    st.set_page_config(layout="wide")
    hide_ui_elements = """
    <style>
    .stActionButton button[kind="header"], .stActionButton div[data-testid="stActionButtonIcon"] {visibility: hidden;}
    .stAppToolbar {display: none;}
    [data-testid="stSidebarCollapsedControl"] {
        width: 70px !important; height: 40px !important;
        background-color: #d2d3d4 !important; border-radius: 15px;
        animation: bounce 2s ease infinite; cursor: pointer;
    }
    @keyframes bounce {0%,100% {transform: translateY(0);} 50% {transform: translateY(-10px);}}
    [data-testid="stSidebarCollapsedControl"]:hover {background-color: #d2d3d4 !important; transform: scale(1.05);}
    html, body, [class*="css"] {font-family: 'Georgia', serif;}
    .title {text-align: center; color: white; font-size: 2.5em;}
    .subheader {text-align: center; color: white; font-size: 1.8em;}
    .caption {text-align: center; color: lightblue; font-size: 1em;}
    </style>
    """
    st.markdown(hide_ui_elements, unsafe_allow_html=True)

def create_header():
    st.markdown(
        f'<div style="display: flex; justify-content: center; margin-top: -50px;">'
        f'<img src="https://raw.githubusercontent.com/ecervera1/st-screener/main/Cervera%20Logo%20BWG.png" width=120>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="title">Portfolio Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Stock Comparative Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">by Eli Cervera</div>', unsafe_allow_html=True)

# ======================
# Core Functions
# ======================
def fetch_stock_data(tickers, start_date, end_date, column='Close'):
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        return data[column] if not data.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def get_financial_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Current Price": info.get("currentPrice"),
            "Market Cap (B)": (info.get("marketCap") or 0) / 1e9,
            "PE Ratio": info.get("trailingPE"),
            "Div Yield": info.get("dividendYield") or 0,
            "52W Range": f"{info.get('fiftyTwoWeekLow', 0):.2f} - {info.get('fiftyTwoWeekHigh', 0):.2f}",
            "Profit Margin": info.get("profitMargins"),
            "ROA": info.get("returnOnAssets"),
            "ROE": info.get("returnOnEquity")
        }
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")
        return {}

def generate_prophet_forecast(ticker, start_date, end_date, forecast_days=365):
    data = fetch_stock_data(ticker, start_date, end_date)
    if data.empty:
        return None

    df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df.dropna())
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    fig = model.plot(forecast)
    plt.title(f'Forecast for {ticker}')
    return fig

def monte_carlo_simulation(data, simulations=1000, days=252):
    returns = data.pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    initial = data.iloc[-1]
    
    final_prices = initial * (1 + np.random.normal(mu, sigma, (days, simulations))).cumprod(axis=0)[-1]
    
    fig, ax = plt.subplots()
    ax.hist(final_prices, bins=50)
    ax.set(xlabel='Price', ylabel='Frequency', title='Monte Carlo Simulation Results')
    stats = f"Mean: {final_prices.mean():.2f}\nMedian: {np.median(final_prices):.2f}\nStd: {final_prices.std():.2f}"
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.8))
    return fig, final_prices

def create_comparison_charts(tickers, start_date, end_date):
    data = fetch_stock_data(tickers, start_date, end_date)
    if data.empty:
        return

    # Performance Charts
    st.title('Stock Performance Chart')
    st.line_chart(data)
    
    # Financial Metrics Table
    st.title('Stock Data')
    metrics = pd.DataFrame({t: get_financial_metrics(t) for t in tickers}).T.fillna('-')
    st.table(metrics)

    # Detailed Comparison Charts
    num_subplots = len(tickers) + 1
    fig, axs = plt.subplots(num_subplots, 5, figsize=(28, num_subplots*4), gridspec_kw={'wspace': 0.5})
    
    # Header Row
    labels = ["Ticker", "Market Cap", "Financial Metrics", "Revenue Comparison", "52-Week Range"]
    for j in range(5):
        axs[0, j].axis('off')
        axs[0, j].text(0.5, 0.5, labels[j], ha='center', va='center', fontsize=25, fontweight='bold')

    for i, ticker in enumerate(tickers, 1):
        # Ticker Label
        axs[i, 0].axis('off')
        axs[i, 0].text(0.5, 0.5, ticker, ha='center', va='center', fontsize=30)

        # Market Cap Visualization
        market_cap = get_financial_metrics(ticker).get("Market Cap (B)", 0) * 1e9
        max_market_cap = max([get_financial_metrics(t).get("Market Cap (B)", 0) * 1e9 for t in tickers])
        ax = axs[i, 1]
        relative_size = (market_cap / max_market_cap) if max_market_cap > 0 else 0
        circle = plt.Circle((0.5, 0.5), relative_size * 0.5, color='lightblue')
        ax.add_artist(circle)
        ax.text(0.5, 0.5, f"{market_cap/1e9:.2f}B", ha='center', va='center', fontsize=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Financial Metrics
        metrics = get_financial_metrics(ticker)
        ax = axs[i, 2]
        values = [metrics.get("Profit Margin", 0)*100, 
                 metrics.get("ROA", 0)*100, 
                 metrics.get("ROE", 0)*100]
        ax.barh(["Profit Margin", "ROA", "ROE"], values, color=['#A3C5A8', '#B8D4B0', '#C8DFBB'])
        ax.axis('off')

        # Revenue Comparison
        stock = yf.Ticker(ticker)
        rev = stock.financials.loc["Total Revenue"].iloc[:2]/1e9
        ax = axs[i, 3]
        ax.bar([0, 1], rev, color=['blue', 'orange'])
        ax.plot([0, 1], rev, color='green' if rev[0] < rev[1] else 'red', marker='o')
        ax.axis('off')

        # 52-Week Range
        prices = get_financial_metrics(ticker)
        ax = axs[i, 4]
        ax.axhline(0.5, xmin=0, xmax=1, color='black', linewidth=3)
        ax.scatter(prices["Current Price"], 0.5, color='red', s=200)
        ax.set_xlim(prices["52W Low"]*0.95, prices["52W High"]*1.05)
        ax.axis('off')

    plt.tight_layout()
    st.pyplot(fig)

# ======================
# Portfolio Functions
# ======================
def portfolio_optimizer(tickers, start_date, end_date, risk_free_rate=0.0):
    data = pd.concat([fetch_stock_data(t, start_date, end_date, 'Adj Close') for t in tickers], axis=1)
    returns = data.pct_change().dropna()
    cov = returns.cov()

    def negative_sharpe(weights):
        port_return = np.dot(weights, returns.mean()) * 252
        port_vol = np.sqrt(weights.T @ cov @ weights) * np.sqrt(252)
        return -(port_return - risk_free_rate) / port_vol

    result = minimize(negative_sharpe, x0=[1/len(tickers)]*len(tickers),
                      bounds=[(0,1)]*len(tickers), constraints={'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    
    if not result.success:
        st.error("Optimization failed: " + result.message)
        return None, None

    return result.x, {
        'Return': np.dot(result.x, returns.mean()) * 252,
        'Volatility': np.sqrt(result.x.T @ cov @ result.x) * np.sqrt(252),
        'Sharpe': -result.fun
    }

# ======================
# Main Application
# ======================
def main():
    configure_page()
    create_header()

    # Sidebar Controls
    with st.sidebar:
        st.title("Controls")
        user_input = st.text_input("Tickers", "LLY, ABT, MRNA, JNJ")
        tickers = [t.strip() for t in user_input.split(',')]
        selected_stock = st.selectbox("Selected Stock", tickers)
        start_date = st.date_input("Start Date", datetime(2021,1,1))
        end_date = st.date_input("End Date", datetime.today())
        
        if st.button('Run Analysis'):
            st.session_state.run_analysis = True
            
        analysis_type = st.radio("Analysis Type", [
            "Performance", 
            "Financials", 
            "Forecasting", 
            "Portfolio",
            "My Portfolio",
            "FinViz Data"
        ])

    # Main Content
    if 'run_analysis' in st.session_state:
        # Performance Analysis
        create_comparison_charts(tickers, start_date, end_date)

        # Financial Statements
        if analysis_type == "Financials":
            st.subheader(f"Financial Statements - {selected_stock}")
            financials = yf.Ticker(selected_stock)
            
            if st.checkbox("Show Income Statement"):
                st.dataframe(financials.financials)
            if st.checkbox("Show Balance Sheet"):
                st.dataframe(financials.balance_sheet)
            if st.checkbox("Show Cash Flow"):
                st.dataframe(financials.cashflow)

        # Forecasting
        elif analysis_type == "Forecasting":
            st.subheader(f"Forecasting - {selected_stock}")
            forecast_days = st.slider("Forecast Period (days)", 30, 365*3, 365)
            
            if st.checkbox("Show Prophet Forecast"):
                fig = generate_prophet_forecast(selected_stock, start_date, end_date, forecast_days)
                if fig: st.pyplot(fig)
            
            if st.checkbox("Run Monte Carlo Simulation"):
                data = fetch_stock_data(selected_stock, start_date, end_date)
                fig, _ = monte_carlo_simulation(data)
                st.pyplot(fig)

        # Portfolio Optimization
        elif analysis_type == "Portfolio":
            st.subheader("Portfolio Optimization")
            risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 0.5) / 100
            
            if st.button("Optimize Portfolio"):
                with st.spinner("Running optimization..."):
                    weights, stats = portfolio_optimizer(
                        tickers, 
                        start_date, 
                        end_date, 
                        risk_free
                    )
                    
                if weights is not None and stats is not None:
                    st.subheader("Optimization Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Optimal Weights")
                        weights_df = pd.DataFrame({
                            'Ticker': tickers,
                            'Weight': weights
                        }).set_index('Ticker')
                        st.dataframe(weights_df.style.format("{:.2%}"))
                        
                    with col2:
                        st.write("### Portfolio Statistics")
                        stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
                        st.dataframe(stats_df.style.format("{:.2f}"))

        # My Portfolio Analysis
        elif analysis_type == "My Portfolio":
            password = st.text_input("Enter Password", type="password")
            if password == "ud":
                # Portfolio analysis implementation
                pass  # Maintain original implementation here

        # FinViz Integration
        elif analysis_type == "FinViz Data":
            # Maintain original FinViz implementation
            pass

        # News Section
        if st.sidebar.checkbox("Show News"):
            st.subheader(f"News - {selected_stock}")
            url = f"https://finance.yahoo.com/quote/{selected_stock}"
            soup = BeautifulSoup(requests.get(url).content, 'html.parser')
            headlines = soup.find_all("h3", class_="Mb(5px)")
            for idx, h in enumerate(headlines[:10]):
                link = f"https://finance.yahoo.com{h.find('a')['href']}"
                st.markdown(f"{idx+1}. [{h.text}]({link})")

if __name__ == "__main__":
    main()
