import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import os

# Carregar modelo de NLP
nlp = spacy.load("pt_core_news_sm")

# Carregar base de dados
caminho_csv = "base_normas_com_recomendacoes_consultas.csv"
df = pd.read_csv(caminho_csv)

# Pré-processamento
def preprocessar(texto):
    doc = nlp(str(texto).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

df["trecho_processado"] = df["trecho"].apply(preprocessar)

# Verificação de base válida
if df["trecho_processado"].dropna().empty:
    st.error("A base de dados está vazia após o pré-processamento. Verifique se há textos válidos no campo 'trecho'.")
    st.stop()

# Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Estilo da página
st.set_page_config(page_title="Diagnóstico por Patologia", layout="centered", initial_sidebar_state="collapsed")

custom_css = """
<style>
    body {
        background-color: #121212;
        color: #f0f0f0;
    }
    .titulo {
        text-align: center;
        font-size: 2.2em;
        font-weight: bold;
        margin-bottom: 0.3em;
        color: #f0f0f0;
    }
    .subtitulo {
        text-align: center;
        font-size: 1.1em;
        color: #c0c0c0;
        margin-bottom: 2em;
    }
    .rodape {
        margin-top: 4em;
        text-align: center;
        font-size: 0.9em;
        color: #c0c0c0;
    }
    .stTextInput label {
        color: #f0f0f0 !important;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Logo
st.image("logo_engenharia.png", use_column_width="auto")

# Título e subtítulo
st.markdown('<div class="titulo">🧱 Diagnóstico por Manifestação Patológica</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</div>', unsafe_allow_html=True)

entrada = st.text_input("Descreva o problema:")

# Função de busca
def buscar_normas(consulta):
    consulta_proc = preprocessar(consulta)
    if not consulta_proc.strip():
        return pd.DataFrame()

    consulta_vec = vetorizador.transform([consulta_proc])
    similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()

    top_indices = similaridades.argsort()[::-1]
    top_resultados = df.iloc[top_indices]
    top_resultados = top_resultados[["manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas"]]
    top_resultados = top_resultados[similaridades[top_indices] > 0.15]  # refinado
    return top_resultados

# Exibição dos resultados
if entrada:
    resultados = buscar_normas(entrada)
    if not resultados.empty:
        st.success("Resultados encontrados:")
        st.dataframe(resultados)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown('<div class="rodape">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
