import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Carrega o modelo de embeddings semânticos
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Lê a base de normas
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Cria coluna combinando manifestação + trecho
df["texto_base"] = df["manifestacao"].fillna("") + ". " + df["trecho"].fillna("")

# Gera embeddings com base no contexto completo
df["embedding"] = df["texto_base"].apply(lambda x: modelo.encode(str(x), convert_to_tensor=True))

# Função de busca por similaridade com filtro por palavra-chave
def buscar_normas(consulta, top_n=5):
    consulta_emb = modelo.encode(consulta, convert_to_tensor=True)
    df["similaridade"] = df["embedding"].apply(lambda emb: util.cos_sim(consulta_emb, emb).item())

    # Tenta priorizar apenas os que contêm a palavra-chave na descrição
    filtrado = df[df["texto_base"].str.lower().str.contains(consulta.lower())]

    if not filtrado.empty:
        resultados = filtrado.sort_values(by="similaridade", ascending=False).head(top_n)
    else:
        resultados = df.sort_values(by="similaridade", ascending=False).head(top_n)

    return resultados[[
        "manifestacao", "norma", "secao", "trecho",
        "recomendacoes", "consultas_relacionadas"
    ]]

# Interface do Streamlit
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
