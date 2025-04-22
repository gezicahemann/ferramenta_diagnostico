import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carrega o modelo de linguagem do spaCy
nlp = spacy.blank("pt")

# Função de pré-processamento
def preprocessar(texto):
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Leitura e preparo da base
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv", encoding="utf-8")
df.columns = df.columns.str.strip()  # Remove espaços extras

# Verificação de colunas obrigatórias
colunas_esperadas = ["manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas"]
for coluna in colunas_esperadas:
    if coluna not in df.columns:
        st.error(f"A coluna '{coluna}' não foi encontrada no arquivo CSV.")
        st.stop()

# Aplica pré-processamento
df["trecho_processado"] = df["trecho"].astype(str).apply(preprocessar)

# Vetorização dos trechos
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Função de busca
def buscar_normas(consulta, top_n=3):
    consulta_proc = preprocessar(consulta)
    consulta_vec = vetorizador.transform([consulta_proc])
    similaridade = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
    indices = similaridade.argsort()[-top_n:][::-1]
    return df.iloc[indices][["manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas"]]

# Configurações de página
st.set_page_config(
    page_title="Diagnóstico Patológico",
    layout="centered",
    page_icon="🧱"
)

# Estilo visual com modo escuro
st.markdown("""
    <style>
        body {
            color: #f0f0f0;
            background-color: #111111;
        }
        .stApp {
            background-color: #111111;
            color: #f0f0f0;
        }
        .title {
            text-align: center;
            font-size: 2.4em;
            font-weight: bold;
            color: #eeeeee;
        }
        .subtitulo {
            text-align: center;
            color: #cccccc;
            font-size: 1.1em;
        }
        .rodape {
            margin-top: 2em;
            text-align: center;
            color: #888888;
            font-size: 0.9em;
        }
    </style>
""", unsafe_allow_html=True)

# Exibe a logo e o título
st.image("logo_engenharia.png", use_column_width=False, width=90)
st.markdown("<div class='title'>🧱 Diagnóstico por Manifestação Patológica</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitulo'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</div>", unsafe_allow_html=True)

# Entrada do usuário
entrada = st.text_input("Descreva o problema:")

# Resultado
if entrada:
    resultados = buscar_normas(entrada)
    st.markdown("## 🔎 Resultados encontrados:")
    for _, linha in resultados.iterrows():
        st.markdown(f"""
        ---
        **Manifestação:** {linha["manifestacao"]}  
        **Norma:** {linha["norma"]} (Seção {linha["secao"]})  
        **Trecho técnico:** {linha["trecho"]}  
        **🔧 Recomendações:** {linha["recomendacoes"]}  
        **📚 Consultas relacionadas:** {linha["consultas_relacionadas"]}
        """)

# Rodapé
st.markdown("<div class='rodape'>Desenvolvido por Gézica Hemann | Engenharia Civil</div>", unsafe_allow_html=True)
