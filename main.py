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
        if data.empty:
            st.error(f"No data found for {tickers} between {start_date} and {end_date}")
            return pd.DataFrame()
        return data[column] if column in data.columns else data['Close']
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return pd.DataFrame()

def get_financial_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Current Price": info.get("currentPrice", 0),
            "Market Cap (B)": (info.get("marketCap", 0) or 0) / 1e9,
            "PE Ratio": info.get("trailingPE", 0),
            "Div Yield": info.get("dividendYield", 0) or 0,
            "52W Low": info.get("fiftyTwoWeekLow", 0),
            "52W High": info.get("fiftyTwoWeekHigh", 0),
            "52W Range": f"{info.get('fiftyTwoWeekLow', 0):.2f} - {info.get('fiftyTwoWeekHigh', 0):.2f}",
            "Profit Margin": info.get("profitMargins", 0),
            "ROA": info.get("returnOnAssets", 0),
            "ROE": info.get("returnOnEquity", 0)
        }
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")
        return {}

def generate_prophet_forecast(ticker, start_date, end_date, forecast_days=365):
    try:
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
    except Exception as e:
        st.error(f"Forecast error: {str(e)}")
        return None

def monte_carlo_simulation(data, simulations=1000, days=252):
    try:
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
    except Exception as e:
        st.error(f"Simulation error: {str(e)}")
        return None, None

def create_comparison_charts(tickers, start_date, end_date):
    try:
        # Current date range data
        data = fetch_stock_data(tickers, start_date, end_date)
        if data.empty:
            return

        # 10-year historical data
        ten_year_start = datetime.now() - timedelta(days=365*10)
        data_10yr = fetch_stock_data(tickers, ten_year_start, end_date)
        
        # Performance Charts
        st.title('Stock Performance Chart')
        st.line_chart(data)
        
        st.title('10-Year Historical Performance')
        st.line_chart(data_10yr)

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
            if prices["52W Low"] > 0 and prices["52W High"] > 0:
                ax.set_xlim(prices["52W Low"]*0.95, prices["52W High"]*1.05)
            else:
                ax.set_xlim(0, data.max()*1.05)
            ax.axis('off')

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Chart error: {str(e)}")

# ======================
# Portfolio Optimization
# ======================
def portfolio_optimizer(tickers, start_date, end_date, risk_free_rate=0.0):
    try:
        if not tickers:
            st.error("Please enter valid tickers")
            return None, None
            
        data = pd.DataFrame()
        valid_tickers = []
        for t in tickers:
            stock_data = fetch_stock_data(t, start_date, end_date, 'Adj Close')
            if not stock_data.empty:
                data[t] = stock_data
                valid_tickers.append(t)
        
        if len(valid_tickers) < 1:
            st.error("No valid data found")
            return None, None
            
        returns = data.pct_change().dropna()
        if returns.empty:
            st.error("Insufficient data for returns")
            return None, None
            
        cov_matrix = returns.cov()
        if cov_matrix.isnull().values.any():
            st.error("Invalid covariance matrix")
            return None, None

        def negative_sharpe(weights):
            port_return = np.dot(weights, returns.mean()) * 252
            port_vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
            return -(port_return - risk_free_rate) / port_vol if port_vol != 0 else -np.inf

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(valid_tickers)))
        initial_guess = [1./len(valid_tickers)] * len(valid_tickers)

        result = minimize(
            negative_sharpe,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        weights = result.x
        stats = {
            'Return': np.dot(weights, returns.mean()) * 252,
            'Volatility': np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252),
            'Sharpe': -result.fun
        }
        
        # Efficient frontier plot
        port_returns = []
        port_volatility = []
        for _ in range(5000):
            weights = np.random.random(len(valid_tickers))
            weights /= np.sum(weights)
            port_returns.append(np.dot(weights, returns.mean()) * 252)
            port_volatility.append(np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252))
            
        plt.figure(figsize=(10, 8))
        plt.scatter(port_volatility, port_returns, 
                   c=(np.array(port_returns)-risk_free_rate)/np.array(port_volatility),
                   cmap='YlGnBu')
        plt.scatter(stats['Volatility'], stats['Return'], 
                   color='red', marker='X', s=200)
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Expected Returns')
        plt.title('Efficient Frontier')
        st.pyplot(plt)
        
        return weights, stats
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return None, None

# ======================
# News & FinViz
# ======================
def show_news(ticker):
    try:
        st.subheader(f"News for {ticker}")
        url = f"https://finance.yahoo.com/quote/{ticker}"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        for h in soup.find_all("h3", class_="Mb(5px)"):
            link = h.find('a')
            if link: 
                news_items.append({
                    "title": link.text,
                    "url": f"https://finance.yahoo.com{link['href']}"
                })
        
        for idx, item in enumerate(news_items[:10]):
            st.markdown(f"{idx+1}. [{item['title']}]({item['url']})")
    except Exception as e:
        st.error(f"News error: {str(e)}")

async def fetch_finviz_data(ticker):
    try:
        async with RetryClient(retry_options=ExponentialRetry(attempts=3)) as client:
            quote = Quote(ticker=ticker)
            return {
                'fundamentals': quote.fundamental_df,
                'news': quote.outer_news_df,
                'insider': quote.insider_trading_df
            }
    except Exception as e:
        st.error(f"FinViz error: {e}")
        return None

def show_finviz_section():
    st.sidebar.title("FinViz Data")
    if st.sidebar.checkbox("Show FinViz Data"):
        ticker = st.sidebar.text_input("Enter ticker:", "AAPL")
        if st.button("Fetch FinViz Data"):
            with st.spinner("Fetching..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                data = loop.run_until_complete(fetch_finviz_data(ticker))
                
                if data:
                    st.subheader("Fundamental Data")
                    st.dataframe(data['fundamentals'])
                    
                    st.subheader("Recent News")
                    st.dataframe(data['news'])
                    
                    st.subheader("Insider Trading")
                    st.dataframe(data['insider'])

# ======================
# Main Application
# ======================
def main():
    configure_page()
    create_header()
    show_finviz_section()

    # Sidebar Controls
    with st.sidebar:
        st.title("Controls")
        user_input = st.text_input("Tickers (comma separated)", "LLY, ABT, MRNA, JNJ")
        tickers = [t.strip() for t in user_input.split(',') if t.strip()]
        selected_stock = st.selectbox("Selected Stock", tickers)
        start_date = st.date_input("Start Date", datetime(2021,1,1))
        end_date = st.date_input("End Date", datetime.today())
        
        analysis_type = st.radio("Analysis Type", [
            "Performance", 
            "Financials", 
            "Forecasting", 
            "Portfolio",
            "News"
        ])
        
        if st.button('Run Analysis'):
            st.session_state.run_analysis = True

    # Main Content
    if 'run_analysis' in st.session_state:
        try:
            if analysis_type == "Performance":
                create_comparison_charts(tickers, start_date, end_date)

            elif analysis_type == "Financials":
                st.subheader(f"Financial Statements - {selected_stock}")
                financials = yf.Ticker(selected_stock)
                
                if st.checkbox("Show Income Statement"):
                    st.dataframe(financials.financials)
                if st.checkbox("Show Balance Sheet"):
                    st.dataframe(financials.balance_sheet)
                if st.checkbox("Show Cash Flow"):
                    st.dataframe(financials.cashflow)

            elif analysis_type == "Forecasting":
                st.subheader(f"Forecasting - {selected_stock}")
                forecast_days = st.slider("Days to forecast", 30, 365*3, 365)
                
                if st.checkbox("Show Prophet Forecast"):
                    fig = generate_prophet_forecast(selected_stock, start_date, end_date, forecast_days)
                    if fig: st.pyplot(fig)
                
                if st.checkbox("Run Monte Carlo Simulation"):
                    data = fetch_stock_data(selected_stock, start_date, end_date)
                    fig, _ = monte_carlo_simulation(data)
                    if fig: st.pyplot(fig)

            elif analysis_type == "Portfolio":
                st.subheader("Portfolio Optimization")
                risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 0.5) / 100
                if st.button("Optimize"):
                    weights, stats = portfolio_optimizer(tickers, start_date, end_date, risk_free)
                    if weights is not None and stats is not None:
                        st.subheader("Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("### Weights")
                            weights_df = pd.DataFrame({
                                'Ticker': tickers,
                                'Weight': weights
                            }).set_index('Ticker')
                            st.dataframe(weights_df.style.format("{:.2%}"))
                        with col2:
                            st.write("### Statistics")
                            stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
                            st.dataframe(stats_df.style.format("{:.2f}"))

            elif analysis_type == "News":
                show_news(selected_stock)

        except Exception as e:
            st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
