import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carrega modelo leve do spaCy (não exige download no Streamlit Cloud)
nlp = spacy.blank("pt")

# Lê a base de normas
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Função de pré-processamento dos textos
def preprocessar(texto):
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Verifica existência da coluna 'trecho' e cria 'trecho_processado'
if "trecho" in df.columns:
    df["trecho_processado"] = df["trecho"].fillna("").apply(preprocessar)
else:
    st.error("Coluna 'trecho' não encontrada no arquivo CSV.")
    st.stop()

# Verifica se há conteúdo útil em 'trecho_processado'
if df["trecho_processado"].isnull().all() or df["trecho_processado"].str.strip().eq("").all():
    st.error("Todos os trechos processados estão vazios. Verifique os dados da coluna 'trecho'.")
    st.stop()

# Vetoriza os textos
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Função para buscar normas mais relevantes
def buscar_normas(consulta, top_n=3):
    consulta_proc = preprocessar(consulta)
    consulta_vec = vetorizador.transform([consulta_proc])
    similaridade = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
    indices = similaridade.argsort()[-top_n:][::-1]
    return df.iloc[indices][[
        "manifestacao", "norma", "trecho", "secao",
        "recomendacoes", "consultas_relacionadas"
    ]]

# Interface no Streamlit
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")
st.title("🧠 Diagnóstico por Manifestação Patológica")
st.markdown("Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)")

entrada = st.text_input("Descreva o problema:")

if entrada:
    resultados = buscar_normas(entrada)
    st.subheader("🔍 Resultados encontrados:")
    for _, linha in resultados.iterrows():
        st.markdown("---")
        st.markdown(f"**Manifestação:** {linha['manifestacao'].capitalize()}")
        st.markdown(f"**Norma:** {linha['norma']} (Seção {linha['secao']})")
        st.markdown(f"**Trecho técnico:** {linha['trecho']}")
        if pd.notna(linha['recomendacoes']):
            st.markdown(f"**🔎 Recomendações de verificação:** {linha['recomendacoes']}")
        if pd.notna(linha['consultas_relacionadas']):
            st.markdown(f"**📚 Sugestões de consulta:** {linha['consultas_relacionadas']}")
