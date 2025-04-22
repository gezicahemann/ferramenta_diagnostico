import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Pré-processamento simples (sem spaCy)
def preprocessar(texto):
    return texto.lower().strip()

# === Carrega a base
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
df.columns = df.columns.str.strip()
df["trecho_processado"] = df["trecho"].astype(str).apply(preprocessar)

# === Remove linhas vazias após pré-processamento
df = df[df["trecho_processado"].str.strip().astype(bool)]

if df.empty:
    st.error("A base de dados está vazia após o pré-processamento.")
    st.stop()

# === Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# === Função de busca
def buscar_normas(consulta, top_n=3):
    consulta_proc = preprocessar(consulta)
    consulta_vec = vetorizador.transform([consulta_proc])
    similaridade = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
    indices = similaridade.argsort()[-top_n:][::-1]
    return df.iloc[indices][[
        "manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas"
    ]]

# === Layout visual
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    .title {
        text-align: center;
        font-size: 2em;
        margin-bottom: 1em;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #888;
        margin-top: 2em;
    }
    </style>
""", unsafe_allow_html=True)

# === Logo
st.image("logo_engenharia.png", width=90)

# === Título
st.markdown("<h1 style='text-align: center;'>🧱 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>", unsafe_allow_html=True)

# === Entrada do usuário
entrada = st.text_input("Descreva o problema:")

# === Resultados
if entrada:
    resultados = buscar_normas(entrada)
    if resultados.empty:
        st.warning("Nenhum resultado encontrado.")
    else:
        st.subheader("🔎 Resultados encontrados:")
        for _, linha in resultados.iterrows():
            st.markdown(f"**Manifestação:** {linha['manifestacao']}")
            st.markdown(f"**Norma:** {linha['norma']} (Seção {linha['secao']})")
            st.markdown(f"**Trecho técnico:** {linha['trecho']}")
            st.markdown(f"**Recomendações:** {linha['recomendacoes']}")
            st.markdown(f"**Consultas relacionadas:** {linha['consultas_relacionadas']}")
            st.markdown("---")

# === Rodapé com seu nome
st.markdown("<div class='footer'>Desenvolvido por Gézica Hemann | Engenharia Civil</div>", unsafe_allow_html=True)
