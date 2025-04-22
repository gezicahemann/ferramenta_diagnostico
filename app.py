import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.pt import Portuguese

# Inicializa NLP leve para tokenização
nlp = Portuguese()

# Função de pré-processamento
def preprocessar(texto):
    doc = nlp(texto.lower())
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Carrega base de dados
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
df["trecho_processado"] = df["trecho"].fillna("").apply(preprocessar)

# Vetoriza os trechos
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Função de busca
def buscar_normas(consulta, top_n=3):
    consulta_proc = preprocessar(consulta)
    consulta_vec = vetorizador.transform([consulta_proc])
    similaridade = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
    indices = similaridade.argsort()[-top_n:][::-1]
    return df.iloc[indices][["manifestacao", "norma", "trecho", "recomendacao"]]

# Layout visual
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

# Estilo com fundo escuro
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #f0f0f0;
    }
    .stApp {
        background-color: #121212;
    }
    .title {
        text-align: center;
        color: #e0e0e0;
        font-size: 2.3em;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #aaa;
        font-size: 1.1em;
    }
    .footer {
        margin-top: 2em;
        text-align: center;
        font-size: 0.9em;
        color: #999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Logo
st.image("logo_engenharia.png", width=100)

# Título e instruções
st.markdown("<div class='title'>🔍 Diagnóstico por Manifestação Patológica</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</div>",
    unsafe_allow_html=True,
)

# Campo de entrada
entrada = st.text_input("Descreva o problema:")

# Resultado
if entrada:
    resultados = buscar_normas(entrada)
    st.subheader("🔎 Resultados encontrados:")
    for _, linha in resultados.iterrows():
        st.markdown(f"**Manifestação:** {linha['manifestacao']}")
        st.markdown(f"**Norma:** {linha['norma']}")
        st.markdown(f"**Trecho técnico:** {linha['trecho']}")
        st.markdown(f"**Recomendações:** {linha['recomendacao']}")
        st.markdown("---")

# Rodapé com seu nome
st.markdown("<div class='footer'>Desenvolvido por Gézica Hemann | Engenharia Civil</div>", unsafe_allow_html=True)
