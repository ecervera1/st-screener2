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
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

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
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(tickers[0], level=1, axis=1) if len(tickers) == 1 else data
        return data[column] if not data.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def scrape_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Current Price": info.get("currentPrice"),
            "Market Cap (B)": (info.get("marketCap") or 0) / 1e9,
            "Profit Margin": info.get("profitMargins"),
            "ROA": info.get("returnOnAssets"),
            "ROE": info.get("returnOnEquity"),
            "52W Range": f"{info.get('fiftyTwoWeekLow', 0):.2f} - {info.get('fiftyTwoWeekHigh', 0):.2f}",
            "52W Low": info.get("fiftyTwoWeekLow"),
            "52W High": info.get("fiftyTwoWeekHigh"),
            "Div Yield": info.get("dividendYield"),
            "Beta": info.get("beta"),
            "Forward Annual Dividend Yield": info.get("dividendYield") or "-",
            "Trailing EPS": info.get("trailingEps"),
            "Forward EPS": info.get("forwardEps"),
            "PE Ratio": info.get("trailingPE"),
            "PEG Ratio": info.get("pegRatio"),
            "Revenue Growth": info.get('revenueGrowth'),
            "Earnings Growth": info.get('earningsGrowth'),
            "Target Low": info.get("targetLowPrice"),
            "Target Mean": info.get("targetMeanPrice"),
            "Recommendation Mean": info.get("recommendationMean"),
        }
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")
        return {}

def generate_prophet_forecast(ticker, start_date, end_date, forecast_days=365):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None

    df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df['y'] = pd.to_numeric(df['y'], errors='coerce').dropna()
    
    if df.empty:
        return None

    model = Prophet()
    model.fit(df)
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
    
    text_x = plt.xlim()[0] + (plt.xlim()[1] - plt.xlim()[0]) * 0.02
    text_y = plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0]) * 0.3
    ax.text(text_x, text_y, f"Simulated Mean: {final_prices.mean():.2f}", fontsize=14)
    ax.text(text_x, text_y*0.9, f"Simulated Median: {np.median(final_prices):.2f}", fontsize=14)
    ax.text(text_x, text_y*0.8, f"Simulated Std: {final_prices.std():.2f}", fontsize=14)
    
    return fig, final_prices

def get_financial_statements(ticker):
    stock = yf.Ticker(ticker)
    return {
        "income_statement": stock.financials,
        "balance_sheet": stock.balance_sheet,
        "cash_flow": stock.cashflow
    }

def calculate_fcff_and_fcfe(ticker):
    stock = yf.Ticker(ticker)
    results = pd.DataFrame()
    
    income = stock.financials
    cashflow = stock.cashflow
    balance = stock.balance_sheet
    
    for i, col in enumerate(income.columns):
        year = pd.to_datetime(col).year
        ni = income.iloc[income.index.str.contains('Net Income', case=False), i].values[0]
        dep = cashflow.iloc[cashflow.index.str.contains('Depreciation', case=False), i].values[0]
        ie = income.iloc[income.index.str.contains('Interest Expense', case=False), i].values[0]
        tax = income.iloc[income.index.str.contains('Tax Provision', case=False), i].values[0]
        capex = cashflow.iloc[cashflow.index.str.contains('Capital Expenditure', case=False), i].values[0]
        
        fcff = ni + dep + ie*(1-tax) - capex
        fcfe = fcff - ie*(1-tax) + (
            cashflow.iloc[cashflow.index.str.contains('Debt Issuance', case=False), i].values[0] -
            cashflow.iloc[cashflow.index.str.contains('Debt Repayment', case=False), i].values[0]
        )
        
        results = pd.concat([results, pd.DataFrame({
            'Year': [year], 'FCFF': [fcff], 'FCFE': [fcfe]
        })])
    
    return results

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

    # Efficient Frontier Plot
    plt.figure(figsize=(10, 8))
    port_returns = []
    port_volatility = []
    for _ in range(5000):
        weights = np.random.random(len(tickers))
        weights /= weights.sum()
        ret = np.dot(weights, returns.mean()) * 252
        vol = np.sqrt(weights.T @ cov @ weights) * np.sqrt(252)
        port_returns.append(ret)
        port_volatility.append(vol)
    
    plt.scatter(port_volatility, port_returns, c=np.array(port_returns)/np.array(port_volatility), cmap='YlGnBu')
    plt.scatter(result.x.T @ cov @ result.x * np.sqrt(252), 
                np.dot(result.x, returns.mean()) * 252, 
                color='red', marker='*', s=500)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    
    return result.x, {
        'Return': np.dot(result.x, returns.mean()) * 252,
        'Volatility': np.sqrt(result.x.T @ cov @ result.x) * np.sqrt(252),
        'Sharpe': -result.fun
    }, plt.gcf()

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
        st.title('Stock Performance Chart')
        data = fetch_stock_data(tickers, start_date, end_date)
        st.line_chart(data)
        
        st.title('10-Year Performance')
        ten_year_start = (datetime.today() - timedelta(days=365*10)).date()
        long_data = fetch_stock_data(tickers, ten_year_start, end_date)
        st.line_chart(long_data)

        # Financial Metrics Table
        st.title('Stock Data')
        metrics_data = pd.DataFrame({t: scrape_stock_data(t) for t in tickers}).T.fillna('-')
        st.dataframe(metrics_data)

        # Detailed Comparison Charts
        fig, axs = plt.subplots(len(tickers)+1, 5, figsize=(28, (len(tickers)+1)*4), gridspec_kw={'wspace': 0.5})
        labels = ["Ticker", "Market Cap", "Financial Metrics", "Revenue Comparison", "52-Week Range"]
        for j in range(5):
            axs[0, j].axis('off')
            axs[0, j].text(0.5, 0.5, labels[j], ha='center', va='center', fontsize=25, fontweight='bold')

        for i, ticker in enumerate(tickers, 1):
            # Ticker Label
            axs[i, 0].axis('off')
            axs[i, 0].text(0.5, 0.5, ticker, ha='center', va='center', fontsize=30)

            # Market Cap Visualization
            market_cap = scrape_stock_data(ticker).get("Market Cap (B)", 0) * 1e9
            max_market_cap = max([scrape_stock_data(t).get("Market Cap (B)", 0) * 1e9 for t in tickers])
            ax = axs[i, 1]
            relative_size = (market_cap / max_market_cap) if max_market_cap > 0 else 0
            circle = plt.Circle((0.5, 0.5), relative_size * 0.5, color='lightblue')
            ax.add_artist(circle)
            ax.text(0.5, 0.5, f"{market_cap/1e9:.2f}B", ha='center', va='center', fontsize=20)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            # Financial Metrics
            metrics = scrape_stock_data(ticker)
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
            prices = scrape_stock_data(ticker)
            ax = axs[i, 4]
            ax.axhline(0.5, xmin=0, xmax=1, color='black', linewidth=3)
            ax.scatter(prices["Current Price"], 0.5, color='red', s=200)
            ax.set_xlim(prices["52W Low"]*0.95, prices["52W High"]*1.05)
            ax.axis('off')

        plt.tight_layout()
        st.pyplot(fig)

        # Financial Statements
        if analysis_type == "Financials":
            st.subheader(f"Financial Statements - {selected_stock}")
            financials = get_financial_statements(selected_stock)
            
            if st.checkbox("Show Income Statement"):
                st.dataframe(financials['income_statement'])
            if st.checkbox("Show Balance Sheet"):
                st.dataframe(financials['balance_sheet'])
            if st.checkbox("Show Cash Flow"):
                st.dataframe(financials['cash_flow'])
            if st.checkbox("Show FCFF/FCFE Analysis"):
                fcff_fcfe = calculate_fcff_and_fcfe(selected_stock)
                st.dataframe(fcff_fcfe)

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
            if st.button("Optimize"):
                weights, stats, plot = portfolio_optimizer(tickers, start_date, end_date, risk_free)
                if weights is not None:
                    st.write("Optimal Weights:", dict(zip(tickers, weights)))
                    st.write("Portfolio Statistics:", stats)
                    st
