import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# === Carrega modelo semântico
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# === Carrega a base
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
df.columns = df.columns.str.strip()

# Combina manifestação + trecho para dar contexto ao embedding
df["texto_base"] = df["manifestacao"].fillna("") + ". " + df["trecho"].fillna("")

# Cria vetores com IA
df["embedding"] = df["texto_base"].apply(lambda x: modelo.encode(str(x), convert_to_tensor=True))

# Função de busca semântica
def buscar_normas(consulta, top_n=3):
    consulta_emb = modelo.encode(consulta, convert_to_tensor=True)
    df["similaridade"] = df["embedding"].apply(lambda emb: util.cos_sim(consulta_emb, emb).item())
    resultados = df.sort_values(by="similaridade", ascending=False).head(top_n)
    return resultados[[
        "manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas", "similaridade"
    ]]

# === Layout
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

# Estilo escuro
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #f0f0f0;
    }
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    .title {
        text-align: center;
        font-size: 2.4em;
        margin-bottom: 1em;
    }
    .footer {
        margin-top: 2em;
        text-align: center;
        font-size: 0.9em;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)

# === Logo e título
st.image("logo_engenharia.png", width=90)
st.markdown("<h1 style='text-align: center;'>🧱 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>", unsafe_allow_html=True)

# === Entrada
entrada = st.text_input("Descreva o problema:")

# === Resultados
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
            st.markdown(f"**💡 Similaridade:** {linha['similaridade']:.2f}")
            st.markdown("---")

# === Rodapé
st.markdown("<div class='footer'>Desenvolvido por Gézica Hemann | Engenharia Civil</div>", unsafe_allow_html=True)
