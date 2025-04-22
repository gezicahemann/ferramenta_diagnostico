import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Estilo da página ===
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

# === Estilização com modo escuro e elementos visuais ===
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #f0f0f0;
        }
        .stApp {
            background-color: #121212;
        }
        h1, h2, h3, h4, h5 {
            color: #f0f0f0;
        }
        .titulo {
            text-align: center;
            font-size: 2.5em;
            font-weight: 600;
            margin-top: 10px;
        }
        .assinatura {
            text-align: center;
            font-size: 0.9em;
            margin-top: 25px;
            color: #888;
        }
        .logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 80px;
            margin-bottom: 10px;
        }
        .stTextInput>div>div>input {
            background-color: #2b2b2b;
            color: white;
        }
        .resultado {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.2rem;
        }
    </style>
""", unsafe_allow_html=True)

# === Logo da engenharia civil ===
st.markdown('<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Engenharia.png/600px-Engenharia.png" class="logo">', unsafe_allow_html=True)

# === Título ===
st.markdown('<div class="titulo">🧱 Diagnóstico por Manifestação Patológica</div>', unsafe_allow_html=True)

st.markdown("Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)")

# === Carrega base de dados ===
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# === Preprocessamento ===
def preprocessar(texto):
    if pd.isna(texto):
        return ""
    return texto.lower()

df["trecho_processado"] = df["trecho"].apply(preprocessar)

# Verifica se há dados válidos para vetorização
trechos_validos = df["trecho_processado"].dropna().astype(str)
trechos_validos = trechos_validos[trechos_validos.str.strip() != ""]

if len(trechos_validos) == 0:
    st.error("Erro: A base de dados está vazia ou mal formatada.")
    st.stop()

# === Vetorização ===
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(trechos_validos)

# === Função de busca ===
def buscar_normas(consulta, top_n=3):
    consulta_proc = preprocessar(consulta)
    consulta_vec = vetorizador.transform([consulta_proc])
    similaridade = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
    indices = similaridade.argsort()[-top_n:][::-1]
    return df.iloc[indices][["manifestacao", "norma", "trecho", "secao", "recomendacao", "consultas"]]

# === Entrada do usuário ===
entrada = st.text_input("Descreva o problema:")

if entrada:
    resultados = buscar_normas(entrada)
    st.subheader("🔎 Resultados encontrados:")

    for _, linha in resultados.iterrows():
        st.markdown(f"""
            <div class="resultado">
                <strong>🔧 Manifestação:</strong> {linha['manifestacao']}<br>
                <strong>📘 Norma:</strong> {linha['norma']} (Seção {linha['secao']})<br>
                <strong>📌 Trecho técnico:</strong><br> {linha['trecho']}<br><br>
                <strong>📋 Recomendações:</strong><br> {linha['recomendacao']}<br><br>
                <strong>🧭 Sugestões de verificação:</strong><br> {linha['consultas']}
            </div>
        """, unsafe_allow_html=True)

# === Assinatura ===
st.markdown('<div class="assinatura">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
