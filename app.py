import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar modelo do spaCy
nlp = spacy.load("pt_core_news_sm")

# Função de pré-processamento
def preprocessar(texto):
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Carregar a base
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Processar os trechos
df["trecho_processado"] = df["trecho"].astype(str).apply(preprocessar)

# Remover linhas vazias
df = df[df["trecho_processado"].str.strip() != ""]

# Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Estilo da página
st.set_page_config(page_title="Diagnóstico por Manifestação Patológica", layout="centered")

st.markdown("""
    <style>
        body {
            background-color: #111111;
            color: #f0f0f0;
        }
        .stTextInput>div>div>input {
            background-color: #222;
            color: white;
        }
        .stTextInput>label {
            color: #dddddd !important;
        }
        .titulo-principal {
            font-size: 2.5em;
            font-weight: bold;
            color: white;
            text-align: center;
        }
        .subtitulo {
            font-size: 1.1em;
            color: #cccccc;
            text-align: center;
        }
        .rodape {
            text-align: center;
            font-size: 0.9em;
            color: #888888;
            margin-top: 2rem;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: -30px;
        }
        .logo-container img {
            width: 80px;
        }
    </style>
""", unsafe_allow_html=True)

# Logo centralizada
st.markdown('<div class="logo-container"><img src="logo_engenharia.png" /></div>', unsafe_allow_html=True)

# Título e subtítulo
st.markdown('<div class="titulo-principal">🔧 Diagnóstico por Manifestação Patológica</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</div>', unsafe_allow_html=True)

# Entrada do usuário
entrada = st.text_input("Descreva o problema:")

def buscar_normas(consulta):
    consulta_proc = preprocessar(consulta)
    if not consulta_proc.strip():
        return pd.DataFrame()

    consulta_vec = vetorizador.transform([consulta_proc])
    similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
    
    top_indices = similaridades.argsort()[::-1]
    top_resultados = df.iloc[top_indices]
    top_resultados = top_resultados[["manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas"]]
    top_resultados = top_resultados[similaridades[top_indices] > 0.1]  # Limite de relevância

    return top_resultados

if entrada:
    resultados = buscar_normas(entrada)
    if not resultados.empty:
        st.success("Resultados encontrados:")
        st.dataframe(resultados)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown('<div class="rodape">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
