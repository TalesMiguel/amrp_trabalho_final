# app.py

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Predição de Séries Temporais Financeiras",
    page_icon="💸",
    layout="wide"
)

st.title("Sistema de Predição de Séries Temporais Financeiras")
st.markdown("Desenvolvido como projeto final da disciplina de Aprendizado de Máquina e Reconhecimento de Padrões da Universidade Federal de São Paulo, 2025/1.")

# ativos do yfinance
ASSETS = {
    # Criptos
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "Solana (SOL)": "SOL-USD",
    "Ripple (XRP)": "XRP-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    #############################
    # indices e commodities
    "S&P 500": "^GSPC",
    "Nasdaq Composite": "^IXIC",
    "Índice de Volatilidade (VIX)": "^VIX",
    # "Índice do Dólar (DXY)": "DX-Y.F", # ta indisponivel ou foi removido do yfinance
    "Ouro": "GC=F",
    "Petróleo Bruto": "CL=F"
}

@st.cache_data # cache maroto pra evitar downloads repetidos
def carregar_dados(tickers, start_date, end_date):
    """
    Baixa os dados financeiros do Yahoo Finance para uma lista de tickers.

    Args:
        tickers (list): lista de tickers de ativos (ex: ['BTC-USD', '^GSPC']).
        start_date (str): data de início no formato 'YYYY-MM-DD'.
        end_date (str): data de fim no formato 'YYYY-MM-DD'.

    Returns:
        pandas.DataFrame: df com os preços de fechamento ajustados ('Adj Close')
                          de todos os ativos solicitados.
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        close_data = data['Close']
        
        if len(tickers) <= 1:
            close_data.columns = [tickers[0]] # Renomeia a coluna para o nome do ticker

        close_data.dropna(inplace=True)

        return close_data

    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

# Define a time window em dias (now - 5*365 = últimos 5 anos)
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Carrega os dados de todos os ativos definidos
todos_os_tickers = list(ASSETS.values())
df_dados = carregar_dados(todos_os_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

if not df_dados.empty:
    st.header("Pré-visualização dos Dados Carregados")
    st.write(f"Exibindo os últimos 5 registros de um total de {len(df_dados)}.")
    st.dataframe(df_dados.tail())
else:
    st.warning("Não foi possível carregar os dados. Verifique a conexão ou os tickers.")
