import streamlit as st
import pandas as pd
import spacy
import pt_core_news_sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregando o modelo do spaCy diretamente
nlp = pt_core_news_sm.load()

# Função para pré-processar os textos
def preprocessar(texto):
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Carregar e preparar a base de dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
    df.dropna(subset=["trecho"], inplace=True)
    df["trecho_processado"] = df["trecho"].astype(str).apply(preprocessar)
    return df

df = carregar_dados()

# Vetorização com TF-IDF
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Função de busca por similaridade
def buscar_normas(consulta):
    consulta_processada = preprocessar(consulta)
    if not consulta_processada.strip():
        return pd.DataFrame()
    vetor_consulta = vetorizador.transform([consulta_processada])
    similaridades = cosine_similarity(vetor_consulta, matriz_tfidf).flatten()
    indices = similaridades.argsort()[::-1][:5]
    resultados = df.iloc[indices][["manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas"]]
    resultados["similaridade"] = similaridades[indices]
    return resultados

# Interface com modo escuro e estilo centralizado
st.markdown(
    """
    <style>
        body {
            color: #fff;
            background-color: #111;
        }
        .stTextInput > label {
            color: #ccc !important;
        }
        .stApp {
            text-align: center;
        }
        .logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo centralizada
st.image("logo_engenharia.png", use_column_width=False, width=80)

# Título e instruções
st.markdown("## 🧱 Diagnóstico por Manifestação Patológica")
st.markdown("Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)")

# Entrada do usuário
entrada = st.text_input("Descreva o problema:")

# Resultado
if entrada:
    resultados = buscar_normas(entrada)
    if resultados.empty:
        st.warning("Nenhum resultado encontrado. Tente descrever de outra forma.")
    else:
        st.success("Resultados encontrados:")
        st.dataframe(resultados.drop(columns=["similaridade"]))

# Rodapé com crédito
st.markdown(
    "<br><div style='text-align: center; color: #888;'>Desenvolvido por Gézica Hemann | Engenharia Civil</div>",
    unsafe_allow_html=True
)
