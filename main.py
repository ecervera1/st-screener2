import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import numpy as np
from prophet import Prophet
from scipy.optimize import minimize
from pyfinviz.quote import Quote
import asyncio
from aiohttp_retry import RetryClient, ExponentialRetry

# ======================
# Configuration & Styling
# ======================
def configure_page():
    st.set_page_config(layout="wide")
    st.markdown("""
    <style>
    .stAppToolbar {display: none;}
    [data-testid="stSidebarCollapsedControl"] {
        width: 70px !important; height: 40px !important;
        background-color: #d2d3d4 !important; 
        border-radius: 15px;
        animation: bounce 2s ease infinite;
    }
    @keyframes bounce {
        0%,100% {transform: translateY(0);}
        50% {transform: translateY(-10px);}
    }
    .news-item {margin-bottom: 10px; font-size: 0.9em;}
    .dataframe {font-size: 0.85em;}
    </style>
    """, unsafe_allow_html=True)

def create_header():
    st.image("https://raw.githubusercontent.com/ecervera1/st-screener/main/Cervera%20Logo%20BWG.png", width=120)
    st.markdown("""
    <div style='text-align: center'>
        <h1 style='color: white'>Portfolio Management</h1>
        <h2 style='color: white'>Stock Comparative Analysis</h2>
        <p style='color: lightblue'>by Eli Cervera</p>
    </div>
    """, unsafe_allow_html=True)

# ======================
# Core Functions
# ======================
def fetch_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        return data.dropna()
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def get_financial_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Price": info.get('currentPrice', '-'),
            "Mkt Cap (B)": f"{(info.get('marketCap', 0)/1e9:.2f}" if info.get('marketCap') else '-',
            "P/E": f"{info.get('trailingPE', '-')}",
            "Div Yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else '-',
            "52W Range": f"{info.get('fiftyTwoWeekLow', '-')} - {info.get('fiftyTwoWeekHigh', '-')}",
            "Profit Margin": f"{info.get('profitMargins', 0)*100:.1f}%" if info.get('profitMargins') else '-'
        }
    except Exception as e:
        st.error(f"Metrics error: {str(e)}")
        return {}

# ======================
# Fixed News Section
# ======================
def show_news(ticker):
    try:
        st.subheader(f"📰 Latest News for {ticker}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        news_items = []
        for article in soup.find_all('h3', {'class': 'Mb(5px)'}):
            link = article.find('a')
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

# ======================
# Fixed FinViz Integration
# ======================
async def fetch_finviz_data(ticker):
    try:
        async with RetryClient(retry_options=ExponentialRetry(attempts=3)) as client:
            quote = Quote(ticker=ticker)
            return {
                'fundamentals': quote.fundamental_df.T,
                'news': quote.outer_news_df.head(5),
                'insider': quote.insider_trading_df.head(5)
            }
    except Exception as e:
        st.error(f"FinViz error: {str(e)}")
        return None

def show_finviz():
    st.sidebar.subheader("FinViz Data")
    ticker = st.sidebar.text_input("Enter ticker for FinViz:", "AAPL")
    if st.sidebar.button("Fetch FinViz Data"):
        with st.spinner("Loading FinViz data..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            data = loop.run_until_complete(fetch_finviz_data(ticker))
            
            if data:
                st.subheader("Fundamental Analysis")
                st.dataframe(data['fundamentals'].style.format("{:.2f}"))
                
                st.subheader("Recent News")
                st.dataframe(data['news'])
                
                st.subheader("Insider Transactions")
                st.dataframe(data['insider'])

# ======================
# Enhanced Charting
# ======================
def create_comparison_charts(tickers, start_date, end_date):
    try:
        # Current data
        data = fetch_stock_data(tickers, start_date, end_date)
        if data.empty:
            return

        # Create main figure
        fig, axs = plt.subplots(len(tickers)+1, 5, figsize=(28, len(tickers)*5))
        
        # Header row
        headers = ["Ticker", "Market Cap", "Financial Metrics", "Revenue Trend", "52W Range"]
        for col in range(5):
            axs[0, col].axis('off')
            axs[0, col].text(0.5, 0.5, headers[col], 
                            ha='center', va='center', 
                            fontsize=20, fontweight='bold')

        for idx, ticker in enumerate(tickers, 1):
            # Ticker label
            axs[idx, 0].axis('off')
            axs[idx, 0].text(0.5, 0.5, ticker, fontsize=24, ha='center', va='center')

            # Market cap bubble chart
            metrics = get_financial_metrics(ticker)
            market_cap = float(metrics["Mkt Cap (B)"].replace('-', '0')) * 1e9
            max_cap = max([float(get_financial_metrics(t)["Mkt Cap (B)"].replace('-', '0')) for t in tickers]) * 1e9
            ax = axs[idx, 1]
            size = (market_cap / max_cap) * 0.5 if max_cap > 0 else 0.1
            circle = plt.Circle((0.5, 0.5), size, color='lightblue')
            ax.add_artist(circle)
            ax.text(0.5, 0.5, metrics["Mkt Cap (B)"], 
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            # Financial metrics bar chart
            ax = axs[idx, 2]
            values = [
                float(metrics["Profit Margin"].replace('%', '').replace('-', '0')),
                float(metrics["ROA"].replace('%', '').replace('-', '0')) if "ROA" in metrics else 0,
                float(metrics["ROE"].replace('%', '').replace('-', '0')) if "ROE" in metrics else 0
            ]
            bars = ax.barh(["Profit", "ROA", "ROE"], values, color=['#4CAF50', '#8BC34A', '#CDDC39'])
            ax.set_xlim(0, max(values)*1.2 if max(values) > 0 else 100)
            ax.bar_label(bars, fmt='%.1f%%', padding=5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Percentage')

            # Revenue trend
            ax = axs[idx, 3]
            stock = yf.Ticker(ticker)
            revenue = stock.financials.loc['Total Revenue'].iloc[:2]/1e9
            years = [str(y.year) for y in revenue.index]
            bars = ax.bar(years, revenue.values, color=['#2196F3', '#64B5F6'])
            ax.bar_label(bars, fmt='%.1fB')
            ax.set_ylabel('Revenue (Billion USD)')
            ax.set_title('Revenue Trend')

            # 52-week range
            ax = axs[idx, 4]
            low = float(metrics["52W Range"].split(' - ')[0].replace('-', '0'))
            high = float(metrics["52W Range"].split(' - ')[-1].replace('-', '0'))
            current = float(metrics["Price"].replace('-', '0'))
            ax.plot([low, high], [0.5, 0.5], color='black', lw=3)
            ax.scatter(current, 0.5, color='red', s=200)
            ax.set_xlim(low*0.95, high*1.05)
            ax.annotate(f'Current: ${current:.2f}', (current, 0.5), 
                       xytext=(0, 20), textcoords='offset points',
                       ha='center', arrowprops=dict(arrowstyle='->'))
            ax.axis('off')

        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Chart error: {str(e)}")

# ======================
# Main Application
# ======================
def main():
    configure_page()
    create_header()
    show_finviz()

    with st.sidebar:
        st.title("Controls")
        tickers = st.text_input("Enter tickers (comma separated)", "LLY, ABT, JNJ").split(',')
        start_date = st.date_input("Start Date", datetime(2021,1,1))
        end_date = st.date_input("End Date", datetime.today())
        analysis_type = st.selectbox("Analysis Type", 
                                   ["Comparison", "Forecast", "News", "Portfolio"])

    if analysis_type == "Comparison":
        create_comparison_charts([t.strip() for t in tickers], start_date, end_date)
    
    elif analysis_type == "News":
        selected = st.selectbox("Select stock for news", [t.strip() for t in tickers])
        show_news(selected)
    
    elif analysis_type == "Forecast":
        selected = st.selectbox("Select stock to forecast", [t.strip() for t in tickers])
        data = fetch_stock_data(selected, start_date, end_date)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prophet Forecast")
            fig = generate_prophet_forecast(selected, start_date, end_date)
            if fig: 
                plt.xlabel('Date')
                plt.ylabel('Price')
                st.pyplot(fig)
        
        with col2:
            st.subheader("Monte Carlo Simulation")
            if not data.empty:
                fig, _ = monte_carlo_simulation(data)
                if fig:
                    plt.xlabel('Price')
                    plt.ylabel('Frequency')
                    st.pyplot(fig)
    
    elif analysis_type == "Portfolio":
        st.subheader("Portfolio Optimizer")
        risk_free = st.number_input("Risk-free rate (%)", 0.0, 10.0, 0.5) / 100
        if st.button("Optimize Portfolio"):
            weights, stats = portfolio_optimizer(
                [t.strip() for t in tickers], 
                start_date, end_date, risk_free
            )
            if weights is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Optimal Weights")
                    weights_df = pd.DataFrame({
                        'Ticker': [t.strip() for t in tickers],
                        'Weight': weights
                    }).set_index('Ticker')
                    st.dataframe(weights_df.style.format("{:.2%}"))
                with col2:
                    st.write("### Portfolio Metrics")
                    metrics_df = pd.DataFrame.from_dict(stats, orient='index')
                    st.dataframe(metrics_df.style.format("{:.2f}"))

if __name__ == "__main__":
    main()
