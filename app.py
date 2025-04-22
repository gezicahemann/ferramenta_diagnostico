import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Carrega o modelo de linguagem em português do spaCy
import subprocess
import importlib.util

model_name = "pt_core_news_sm"
if importlib.util.find_spec(model_name) is None:
    subprocess.run(["python", "-m", "spacy", "download", model_name])

nlp = spacy.load(model_name)

# Lê a base de normas atualizada com recomendações e sugestões
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Pré-processa os textos das normas
def preprocessar(texto):
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

df["trecho_processado"] = df["trecho"].apply(preprocessar)

# Vetoriza os trechos usando TF-IDF
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Função de busca dos trechos mais semelhantes
def buscar_normas(consulta, top_n=3):
    consulta_proc = preprocessar(consulta)
    consulta_vec = vetorizador.transform([consulta_proc])
    similaridade = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
    indices = similaridade.argsort()[-top_n:][::-1]
    return df.iloc[indices][["manifestacao", "norma", "secao", "trecho", "recomendacao", "consultas_tecnicas"]]

# Interface com o usuário no Streamlit
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")
st.title("🧱 Diagnóstico por Manifestação Patológica")
st.markdown("Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)")

entrada = st.text_input("Descreva o problema:")

if entrada:
    resultados = buscar_normas(entrada)
    st.subheader("🔍 Resultados encontrados:")
    for _, linha in resultados.iterrows():
        st.markdown(f"**Manifestação:** {linha['manifestacao'].capitalize()}")
        st.markdown(f"**Norma:** {linha['norma']} (Seção {linha['secao']})")
        st.markdown(f"**Trecho técnico:** {linha['trecho']}")
        if pd.notna(linha['recomendacao']):
            st.markdown(f"**Recomendações de verificação:** {linha['recomendacao']}")
        if pd.notna(linha['consultas_tecnicas']):
            st.markdown(f"**Sugestão de consultas técnicas:** {linha['consultas_tecnicas']}")
        st.markdown("---")
