import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# === Carrega modelo de embeddings
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# === Lê a base
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
df.columns = df.columns.str.strip()

# Combina manifestação e trecho técnico para dar contexto ao embedding
df["texto_base"] = df["manifestacao"].fillna("") + ". " + df["trecho"].fillna("")

# Gera embeddings semânticos com IA
df["embedding"] = df["texto_base"].apply(lambda x: modelo.encode(str(x), convert_to_tensor=True))

# === Função de busca com filtro técnico por palavra-chave
def buscar_normas(consulta, top_n=5):
    consulta_emb = modelo.encode(consulta, convert_to_tensor=True)
    df["similaridade"] = df["embedding"].apply(lambda emb: util.cos_sim(consulta_emb, emb).item())

    # 🔍 Filtro por termo explícito (refina resultados)
    contem_palavra = df["texto_base"].str.lower().str.contains(consulta.lower())
    base_filtrada = df[contem_palavra] if contem_palavra.any() else df

    # Retorna ordenado por similaridade
    resultados = base_filtrada.sort_values(by="similaridade", ascending=False).head(top_n)
    return resultados[[
        "manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas", "similaridade"
    ]]

# === Interface
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

# Estilo escuro com contraste e identidade
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    .footer {
        margin-top: 2em;
        text-align: center;
        color: #888;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Logo
st.image("logo_engenharia.png", width=90)

# Título
st.markdown("<h1 style='text-align: center;'>🧱 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>", unsafe_allow_html=True)

# Entrada do usuário
entrada = st.text_input("Descreva o problema:")

# Resultados com filtro
if entrada:
    resultados = buscar_normas(entrada)
    if resultados.empty:
        st.warning("Nenhum resultado encontrado.")
    else:
        st.subheader("🔎 Resultados encontrados:")
        for _, linha in resultados.iterrows():
            st.markdown(f"**Manifestação:** {linha['manifestacao']}")
            st.markdown(f"**Norma:** {linha['norma']} (Seção {linha['secao']})")
            st.markdown(f"**Trecho técnico:** {linha['trecho']}")
            st.markdown(f"**🔧 Recomendações:** {linha['recomendacoes']}")
            st.markdown(f"**📚 Consultas relacionadas:** {linha['consultas_relacionadas']}")
            st.markdown("---")

# Rodapé com nome
st.markdown("<div class='footer'>Desenvolvido por Gézica Hemann | Engenharia Civil</div>", unsafe_allow_html=True)
