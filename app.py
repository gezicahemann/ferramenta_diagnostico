import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carrega modelo spaCy
nlp = spacy.load("pt_core_news_sm")

# Lê a base de dados
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Pré-processamento com spaCy
def preprocessar(texto):
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

df["trecho_processado"] = df["trecho"].astype(str).apply(preprocessar)

# Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Busca por similaridade
def buscar_normas(consulta, top_n=3):
    consulta_proc = preprocessar(consulta)
    consulta_vec = vetorizador.transform([consulta_proc])
    similaridade = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
    indices = similaridade.argsort()[-top_n:][::-1]
    return df.iloc[indices][["manifestacao", "norma", "trecho", "verificacao"]]

# CONFIGURAÇÃO VISUAL
st.set_page_config(
    page_title="Diagnóstico Patológico",
    layout="centered",
    initial_sidebar_state="auto"
)

# Estilo com modo escuro e tipografia
st.markdown("""
    <style>
        body {
            background-color: #111;
            color: #f0f0f0;
        }
        .stApp {
            background-color: #111;
            color: #f0f0f0;
            font-family: 'Segoe UI', sans-serif;
        }
        .title-style {
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            color: #f0f0f0;
            margin-bottom: 10px;
        }
        .sub-style {
            text-align: center;
            font-size: 16px;
            color: #aaa;
            margin-bottom: 30px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 40px;
            color: #666;
        }
        .stTextInput > div > div > input {
            background-color: #222;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Logo ajustada
st.image("logo_engenharia.png", width=100)

# Título
st.markdown('<div class="title-style">🧱 Diagnóstico por Manifestação Patológica</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-style">Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</div>', unsafe_allow_html=True)

# Entrada do usuário
entrada = st.text_input("Descreva o problema:")

# Resultado
if entrada:
    resultados = buscar_normas(entrada)
    st.subheader("🔎 Resultados encontrados:")
    for _, linha in resultados.iterrows():
        st.markdown(f"**Manifestação:** {linha['manifestacao']}")
        st.markdown(f"**Norma:** {linha['norma']}")
        st.markdown(f"**Trecho técnico:** {linha['trecho']}")
        st.markdown(f"**Recomendações de verificação:** {linha['verificacao']}")
        st.markdown("---")

# Rodapé com nome
st.markdown('<div class="footer">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
