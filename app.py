import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carrega modelo leve do spaCy para português
nlp = spacy.blank("pt")

# Função de pré-processamento
def preprocessar(texto):
    doc = nlp(texto.lower())
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    return " ".join(tokens)

# Carrega base de dados
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Preprocessa os trechos
df["trecho_processado"] = df["trecho"].astype(str).apply(preprocessar)

# Remove linhas vazias após o processamento
df = df[df["trecho_processado"].str.strip().astype(bool)]

# Verifica se a base está válida
if df.empty:
    st.error("A base de dados está vazia após o pré-processamento. Verifique se há textos válidos na coluna 'trecho'.")
    st.stop()

# Cria matriz TF-IDF
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Função de busca
def buscar_normas(consulta, top_n=5):
    consulta_processada = preprocessar(consulta)
    vetor_consulta = vetorizador.transform([consulta_processada])
    similaridades = cosine_similarity(vetor_consulta, matriz_tfidf).flatten()
    indices_top = similaridades.argsort()[-top_n:][::-1]
    return df.iloc[indices_top][["manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas"]]

# Estilização da interface
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered", page_icon="🧱")

# Interface escura customizada
st.markdown("""
    <style>
        body {
            background-color: #111;
            color: #f1f1f1;
        }
        .stTextInput label {
            color: #f1f1f1;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Logo e título centralizados
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("logo_engenharia.png", width=80)

st.markdown("<h1 style='text-align: center;'>🧱 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>", unsafe_allow_html=True)

# Campo de entrada
entrada = st.text_input("Descreva o problema:")

# Busca e exibição dos resultados
if entrada:
    resultados = buscar_normas(entrada)
    if resultados.empty:
        st.warning("Nenhuma correspondência encontrada.")
    else:
        st.success("Resultados encontrados:")
        st.dataframe(resultados, use_container_width=True)

# Rodapé
st.markdown("<p style='text-align: center; margin-top: 2rem;'>Desenvolvido por Gézica Hemann | Engenharia Civil</p>", unsafe_allow_html=True)
