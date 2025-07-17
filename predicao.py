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

# dicionario invertido para buscar nome pelo ticker
ASSETS_INV = {v: k for k, v in ASSETS.items()}


@st.cache_data # cache maroto pra evitar downloads repetidos
def carregar_dados(tickers, start_date, end_date):
    """
    Baixa os dados financeiros do Yahoo Finance para uma lista de tickers.

    Args:
        tickers (list): lista de tickers de ativos (ex: ['BTC-USD', '^GSPC']).
        start_date (str): data de inicio no formato 'YYYY-MM-DD'.
        end_date (str): data de fim no formato 'YYYY-MM-DD'.

    returns:
        pandas.DataFrame: df com os precos de fechamento ('Close')
                          de todos os ativos solicitados.
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.error("nenhum dado retornado. verifique o periodo ou os tickers.")
            return pd.DataFrame()

        close_data = data['Close']
        if len(tickers) == 1:
            close_data = pd.DataFrame(close_data)
            close_data.columns = [tickers[0]]
            
        # dropna remove linhas com qualquer valor NaN
        close_data.dropna(axis=0, how='any', inplace=True)
        return close_data
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

def criar_features_e_alvo(df, ticker_alvo, tamanho_janela, horizonte):
    """
    Cria features usando a tecnica de janelamento e o alvo da predicao
    """
    X, y = [], []
    
    # .shift(-horizonte) puxa os valores do futuro para a linha atual, criando o alvo (y)
    y_completo = df[ticker_alvo].shift(-horizonte)

    for i in range(len(df) - tamanho_janela - horizonte + 1):
        janela = df.iloc[i : i + tamanho_janela]
        
        # .flatten() transforma a janela de dados em um vetor de features
        features = janela.values.flatten()
        X.append(features)
        
        alvo_idx = i + tamanho_janela + horizonte - 1
        y.append(y_completo.iloc[alvo_idx])

    X = pd.DataFrame(X)
    y = pd.Series(y)

    X.dropna(inplace=True)
    y.dropna(inplace=True)
    
    # .intersection() alinha X e y para garantir que tenham os mesmos indices
    indices_comuns = X.index.intersection(y.index)
    X = X.loc[indices_comuns]
    y = y.loc[indices_comuns]
    
    return X, y

st.sidebar.header("Configuracoes da Predicao")

ativo_alvo_nome = st.sidebar.selectbox(
    "Selecione a Criptomoeda Alvo:",
    options=list(ASSETS.keys()),
    index=0 # BTC como padrao
)
ticker_alvo = ASSETS[ativo_alvo_nome]

nomes_ativos_auxiliares = st.sidebar.multiselect(
    "Selecione os Ativos Auxiliares:",
    options=list(ASSETS.keys()),
    default=["Ethereum (ETH)", "S&P 500"] # opcoes padrao
)
tickers_auxiliares = [ASSETS[nome] for nome in nomes_ativos_auxiliares]

# configs de janelamento e horizonte
st.sidebar.subheader("Configurações de Janelamento e Horizonte")
tamanho_janela = st.sidebar.slider("Tamanho da Janela (dias para tras):", 1, 60, 10)
horizonte_predicao = st.sidebar.selectbox("Horizonte de Predicao (dias a frente):", [1, 3, 5, 7], index=0)

tipo_saida = st.sidebar.selectbox(
    "Selecione o Tipo de Saida:",
    ["Valor da Serie (Regressao)", "Subida/Descida (Classificacao)"]
)

# define a timewindow em dias (hoje - dias * dias_em_um_ano)
end_date = datetime.now()
start_date = end_date - timedelta(days=10*365)

tickers_selecionados = [ticker_alvo] + tickers_auxiliares
df_dados = carregar_dados(tickers_selecionados, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

if not df_dados.empty:
    st.header(f"Dados Históricos para {ativo_alvo_nome} e Auxiliares")
    st.line_chart(df_dados)

    st.subheader("Dados Brutos")
    st.dataframe(df_dados.tail())
    
    # geracao de features e alvo
    st.header("Engenharia de Features (Janelamento)")
    X, y_regressao = criar_features_e_alvo(df_dados, ticker_alvo, tamanho_janela, horizonte_predicao)

    if tipo_saida == "Subida/Descida (Classificação)":
        # if preco_atual > preco_futuro: 1; else 0
        # ultimo preco da janela de cada ativo alvo
        preco_atual = df_dados[ticker_alvo].loc[X.index + tamanho_janela + horizonte_predicao - 1].values
        y = pd.Series((y_regressao.values > preco_atual), dtype=int)
        y.index = X.index
    else:
        y = y_regressao

    if not X.empty and not y.empty:
        st.write(f"Foram geradas **{X.shape[0]}** amostras de treinamento.")
        st.write(f"Cada amostra tem **{X.shape[1]}** features (janela de {tamanho_janela} dias x {len(tickers_selecionados)} ativos).")
        
        st.subheader("Amostra das Features (X)")
        st.dataframe(X.head())

        st.subheader("Amostra do Alvo (y)")
        st.dataframe(y.head())
    else:
        st.warning("Não foi possível gerar as features. Tente um período maior ou janela menor.")

else:
    st.warning("Não foi possível carregar os dados. Verifique a conexão ou os tickers.")
