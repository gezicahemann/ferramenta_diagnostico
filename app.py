import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Carrega o modelo de linguagem leve com embeddings
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Lê a base de normas
df = pd.read_csv("base_normas_streamlit.csv")

# Gera os embeddings dos trechos da base
df["embedding"] = df["trecho"].apply(lambda x: modelo.encode(x, convert_to_tensor=True))

# Função para buscar os trechos mais semelhantes semanticamente
def buscar_normas(consulta, top_n=3):
    consulta_embedding = modelo.encode(consulta, convert_to_tensor=True)
    similares = [util.cos_sim(consulta_embedding, emb).item() for emb in df["embedding"]]
    df["similaridade"] = similares
    resultados = df.sort_values(by="similaridade", ascending=False).head(top_n)
    return resultados[["manifestacao", "norma", "trecho", "secao"]]

# Interface no Streamlit
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
        st.markdown("---")
