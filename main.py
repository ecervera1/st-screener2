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
    .news-item {margin-bottom: 10px; font-size: 0.9em;}
    .dataframe {font-size: 0.85em; margin: 0 auto;}
    .dataframe th {padding: 0.5em !important;}
    .dataframe td {padding: 0.5em !important;}
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
            st.error(f"No data found for {tickers}")
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
            "Price": info.get("currentPrice", 0),
            "Mkt Cap (B)": round((info.get("marketCap", 0) or 0) / 1e9, 2),  # Fixed line
            "P/E": round(info.get("trailingPE", 0), 1),
            "Div Yield": f"{round(info.get('dividendYield', 0)*100, 2)}%" if info.get('dividendYield') else '-',
            "52W Low": info.get("fiftyTwoWeekLow", 0),
            "52W High": info.get("fiftyTwoWeekHigh", 0),
            "Profit Margin": f"{round(info.get('profitMargins', 0)*100, 1)}%" if info.get('profitMargins') else '-',
            "ROA": f"{round(info.get('returnOnAssets', 0)*100, 1)}%" if info.get('returnOnAssets') else '-',
            "ROE": f"{round(info.get('returnOnEquity', 0)*100, 1)}%" if info.get('returnOnEquity') else '-'
        }
    except Exception as e:
        st.error(f"Metrics error: {e}")
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
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(final_prices, bins=50)
        ax.set(xlabel='Price', ylabel='Frequency', title='Monte Carlo Simulation')
        stats = f"Mean: {final_prices.mean():.2f}\nMedian: {np.median(final_prices):.2f}\nStd Dev: {final_prices.std():.2f}"
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, va='top', 
                bbox=dict(facecolor='white', alpha=0.8))
        return fig, final_prices
    except Exception as e:
        st.error(f"Simulation error: {str(e)}")
        return None, None

# ======================
# Comparison Charts
# ======================
def create_comparison_charts(tickers, start_date, end_date):
    try:
        # Current data
        data = fetch_stock_data(tickers, start_date, end_date)
        if data.empty:
            return

        # 10-year data
        ten_year_start = datetime.now() - timedelta(days=365*10)
        data_10yr = fetch_stock_data(tickers, ten_year_start, end_date)
        
        st.title('Price Performance')
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Selected Period")
            st.line_chart(data)
        with col2:
            st.write("#### 10-Year History")
            st.line_chart(data_10yr)

        # Financial metrics
        st.title('Financial Metrics')
        metrics = pd.DataFrame({t: get_financial_metrics(t) for t in tickers}).T
        st.table(metrics.style.set_properties(**{'text-align': 'center'}))

        # Detailed charts
        st.title('Detailed Analysis')
        num_subplots = len(tickers) + 1
        fig, axs = plt.subplots(num_subplots, 5, figsize=(28, num_subplots*4), gridspec_kw={'wspace': 0.5})
        
        # Header row
        labels = ["Ticker", "Market Cap", "Metrics", "Revenue", "52W Range"]
        for j in range(5):
            axs[0, j].axis('off')
            axs[0, j].text(0.5, 0.5, labels[j], ha='center', va='center', 
                          fontsize=22, fontweight='bold')

        for i, ticker in enumerate(tickers, 1):
            # Ticker label
            axs[i, 0].axis('off')
            axs[i, 0].text(0.5, 0.5, ticker, ha='center', va='center', fontsize=26)

            # Market cap bubble
            metrics = get_financial_metrics(ticker)
            market_cap = metrics["Mkt Cap (B)"] * 1e9
            max_mcap = max([get_financial_metrics(t)["Mkt Cap (B)"] * 1e9 for t in tickers])
            ax = axs[i, 1]
            rel_size = (market_cap / max_mcap) if max_mcap > 0 else 0.1
            circle = plt.Circle((0.5, 0.5), rel_size*0.5, color='lightblue')
            ax.add_artist(circle)
            ax.text(0.5, 0.5, f"{metrics['Mkt Cap (B)']:.1f}B", ha='center', va='center', fontsize=18)
            ax.axis('off')

            # Financial metrics
            ax = axs[i, 2]
            values = [float(metrics["Profit Margin"].strip('%')) if metrics["Profit Margin"] != '-' else 0,
                     float(metrics["ROA"].strip('%')) if metrics["ROA"] != '-' else 0,
                     float(metrics["ROE"].strip('%')) if metrics["ROE"] != '-' else 0]
            ax.barh(["Profit", "ROA", "ROE"], values, color=['#A3C5A8', '#B8D4B0', '#C8DFBB'])
            ax.axis('off')

            # Revenue comparison
            stock = yf.Ticker(ticker)
            rev = stock.financials.loc["Total Revenue"].iloc[:2]/1e9
            ax = axs[i, 3]
            ax.bar([0, 1], rev, color=['blue', 'orange'])
            ax.plot([0, 1], rev, color='green' if rev[0] < rev[1] else 'red', marker='o')
            ax.axis('off')

            # 52-week range
            ax = axs[i, 4]
            current_price = metrics["Price"]
            low = metrics["52W Low"]
            high = metrics["52W High"]
            ax.axhline(0.5, xmin=0, xmax=1, color='black', linewidth=3)
            ax.scatter(current_price, 0.5, color='red', s=200)
            if low > 0 and high > 0:
                ax.set_xlim(low*0.95, high*1.05)
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
            st.error("Enter valid tickers")
            return None, None
            
        data = pd.DataFrame()
        valid_tickers = []
        for t in tickers:
            stock_data = fetch_stock_data(t, start_date, end_date, 'Adj Close')
            if not stock_data.empty:
                data[t] = stock_data
                valid_tickers.append(t)
        
        if len(valid_tickers) < 2:
            st.error("Need at least 2 valid tickers")
            return None, None
            
        returns = data.pct_change().dropna()
        if returns.empty:
            st.error("Insufficient data")
            return None, None
            
        cov_matrix = returns.cov()
        if cov_matrix.isnull().values.any():
            st.error("Invalid data")
            return None, None

        def negative_sharpe(weights):
            port_return = np.dot(weights, returns.mean()) * 252
            port_vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
            return -(port_return - risk_free_rate) / port_vol if port_vol != 0 else -np.inf

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in valid_tickers)
        initial_guess = [1./len(valid_tickers)] * len(valid_tickers)

        result = minimize(negative_sharpe, initial_guess, 
                        method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        weights = result.x
        stats = {
            'Return': np.dot(weights, returns.mean()) * 252,
            'Volatility': np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252),
            'Sharpe': -result.fun
        }
        
        # Efficient frontier
        port_returns = []
        port_volatility = []
        for _ in range(5000):
            w = np.random.random(len(valid_tickers))
            w /= w.sum()
            port_returns.append(np.dot(w, returns.mean()) * 252)
            port_volatility.append(np.sqrt(w.T @ cov_matrix @ w) * np.sqrt(252))
            
        plt.figure(figsize=(10, 6))
        plt.scatter(port_volatility, port_returns, 
                   c=(np.array(port_returns)-risk_free_rate)/np.array(port_volatility),
                   cmap='YlGnBu', alpha=0.5)
        plt.scatter(stats['Volatility'], stats['Return'], 
                   color='red', marker='X', s=200, label='Optimal')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
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
        st.subheader(f"📰 Latest News for {ticker}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        for item in soup.find_all('h3', class_='Mb(5px)'):
            link = item.find('a')
            if link and link.text.strip():
                news_items.append({
                    'title': link.text.strip(),
                    'url': f"https://finance.yahoo.com{link['href']}"
                })
        
        for idx, item in enumerate(news_items[:5]):
            st.markdown(f"""
            <div class="news-item">
            {idx+1}. <a href="{item['url']}" target="_blank">{item['title']}</a>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"News error: {str(e)}")

async def fetch_finviz_data(ticker):
    try:
        async with RetryClient(retry_options=ExponentialRetry(attempts=3)) as client:
            quote = Quote(ticker=ticker)
            return {
                'fundamentals': quote.fundamental_df.T.style.format("{:.2f}"),
                'news': quote.outer_news_df.head(5),
                'insider': quote.insider_trading_df.head(5)
            }
    except Exception as e:
        st.error(f"FinViz error: {e}")
        return None

def show_finviz_section():
    st.sidebar.title("FinViz Data")
    if st.sidebar.checkbox("Show FinViz Data"):
        ticker = st.sidebar.text_input("Ticker:", "AAPL")
        if st.button("Fetch"):
            with st.spinner("Loading..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                data = loop.run_until_complete(fetch_finviz_data(ticker))
                
                if data:
                    st.subheader("Fundamentals")
                    st.table(data['fundamentals'])
                    
                    st.subheader("Recent News")
                    st.table(data['news'])
                    
                    st.subheader("Insider Activity")
                    st.table(data['insider'])

# ======================
# Main Application
# ======================
def main():
    configure_page()
    create_header()
    show_finviz_section()

    with st.sidebar:
        st.title("Controls")
        user_input = st.text_input("Tickers (comma separated)", "LLY, ABT, MRNA, JNJ")
        tickers = [t.strip() for t in user_input.split(',') if t.strip()]
        selected_stock = st.selectbox("Selected Stock", tickers)
        start_date = st.date_input("Start Date", datetime(2021,1,1))
        end_date = st.date_input("End Date", datetime.today())
        
        analysis_type = st.radio("Analysis", [
            "Performance", 
            "Financials", 
            "Forecasting", 
            "Portfolio",
            "News"
        ])
        
        if st.button('Run Analysis'):
            st.session_state.run_analysis = True

    if 'run_analysis' in st.session_state:
        try:
            if analysis_type == "Performance":
                create_comparison_charts(tickers, start_date, end_date)

            elif analysis_type == "Financials":
                st.subheader(f"Financials - {selected_stock}")
                financials = yf.Ticker(selected_stock)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.checkbox("Income Statement"):
                        st.table(financials.financials.style.format("{:.0f}"))
                with col2:
                    if st.checkbox("Balance Sheet"):
                        st.table(financials.balance_sheet.style.format("{:.0f}"))
                if st.checkbox("Cash Flow"):
                    st.table(financials.cashflow.style.format("{:.0f}"))

            elif analysis_type == "Forecasting":
                st.subheader(f"Forecasting - {selected_stock}")
                forecast_days = st.slider("Forecast Horizon (days)", 30, 365*3, 365)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.checkbox("Show Prophet Forecast"):
                        fig = generate_prophet_forecast(selected_stock, start_date, end_date, forecast_days)
                        if fig: st.pyplot(fig)
                with col2:
                    if st.checkbox("Run Monte Carlo"):
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
                            st.write("### Allocation")
                            alloc = pd.DataFrame({
                                'Ticker': tickers,
                                'Weight': weights
                            }).set_index('Ticker')
                            st.table(alloc.style.format("{:.1%}"))
                        with col2:
                            st.write("### Statistics")
                            stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
                            st.table(stats_df.style.format("{:.2f}"))

            elif analysis_type == "News":
                show_news(selected_stock)

        except Exception as e:
            st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
