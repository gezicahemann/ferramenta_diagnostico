import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carrega o modelo de linguagem
nlp = spacy.blank("pt")

# Função de pré-processamento
def preprocessar(texto):
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Carrega a base
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Aplica pré-processamento
df["trecho_processado"] = df["trecho"].astype(str).apply(preprocessar)

# Remove linhas com trechos vazios após o processamento
df = df[df["trecho_processado"].str.strip().astype(bool)]

# Verifica se a base está vazia após o processamento
if df.empty:
    st.error("A base de dados está vazia após o pré-processamento. Verifique se há textos válidos no campo 'trecho'.")
    st.stop()

# Vetoriza os trechos
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Interface
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

# Estilo escuro com contraste
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
    </style>
""", unsafe_allow_html=True)

# Logo
st.image("logo_engenharia.png", width=100)

# Título
st.markdown("<h1 style='text-align: center; color: #f5f5f5;'>🧱 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)

# Subtítulo
st.markdown("<p style='text-align: center; color: #b0b0b0;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>", unsafe_allow_html=True)

# Entrada do usuário
entrada = st.text_input("Descreva o problema:")

def buscar_normas(texto_usuario):
    texto_processado = preprocessar(texto_usuario)
    vetor_entrada = vetorizador.transform([texto_processado])
    similaridades = cosine_similarity(vetor_entrada, matriz_tfidf).flatten()
    indices = similaridades.argsort()[::-1]
    resultados = df.iloc[indices][["manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas"]]
    resultados = resultados[similaridades[indices] > 0.1]  # Filtra por similaridade mínima
    return resultados

# Exibe resultados
if entrada:
    resultados = buscar_normas(entrada)
    if resultados.empty:
        st.warning("Nenhum resultado encontrado para essa manifestação.")
    else:
        st.markdown("### 🔍 Resultados encontrados:")
        for _, row in resultados.iterrows():
            st.markdown(f"**Manifestação:** {row['manifestacao']}")
            st.markdown(f"**Norma:** {row['norma']} (Seção {row['secao']})")
            st.markdown(f"**Trecho técnico:** {row['trecho']}")
            st.markdown(f"**Recomendações:** {row['recomendacoes']}")
            st.markdown(f"**Consultas relacionadas:** {row['consultas_relacionadas']}")
            st.markdown("---")

# Rodapé
st.markdown("<p style='text-align: center; margin-top: 2em; color: #888;'>Desenvolvido por Gézica Hemann | Engenharia Civil</p>", unsafe_allow_html=True)
