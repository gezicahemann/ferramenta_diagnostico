import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Estilo escuro com contraste agradável
st.set_page_config(
    page_title="Diagnóstico Patológico",
    layout="centered",
    initial_sidebar_state="auto"
)

# Aplica CSS para fundo escuro e letras mais visíveis
st.markdown(
    """
    <style>
        body {
            background-color: #111111;
            color: #f0f0f0;
        }
        .stApp {
            background-color: #111111;
            color: #f0f0f0;
            font-family: 'Segoe UI', sans-serif;
        }
        .title-style {
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            color: #f0f0f0;
            margin-bottom: 10px;
        }
        .sub-style {
            text-align: center;
            font-size: 16px;
            color: #999;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 40px;
            color: #666;
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

# Exibe logo
st.image("logo_engenharia.png", use_column_width=False, width=80)

# Título e instrução
st.markdown('<div class="title-style">🔎 Diagnóstico por Manifestação Patológica</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-style">Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</div>',
    unsafe_allow_html=True
)

# Carrega base de dados
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Pré-processamento
def preprocessar(texto):
    return texto.lower().strip()

df["trecho_processado"] = df["trecho"].astype(str).apply(preprocessar)

# Verifica se há dados para processar
if df["trecho_processado"].isnull().all() or df["trecho_processado"].str.strip().eq("").all():
    st.error("Erro: Base de dados não possui trechos válidos para análise.")
else:
    # Vetorização
    vetorizador = TfidfVectorizer()
    matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

    # Busca inteligente
    def buscar_normas(consulta, top_n=3):
        consulta_proc = preprocessar(consulta)
        consulta_vec = vetorizador.transform([consulta_proc])
        similaridade = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
        indices = similaridade.argsort()[-top_n:][::-1]
        return df.iloc[indices][["manifestacao", "norma", "trecho", "secao", "recomendacao"]]

    entrada = st.text_input("Descreva o problema:")

    if entrada:
        resultados = buscar_normas(entrada)
        st.subheader("📄 Resultados encontrados:")

        for _, linha in resultados.iterrows():
            st.markdown(f"**Manifestação:** {linha['manifestacao']}")
            st.markdown(f"**Norma:** {linha['norma']} (Seção {linha['secao']})")
            st.markdown(f"**Trecho técnico:** {linha['trecho']}")
            st.markdown(f"**Recomendações de verificação:** {linha['recomendacao']}")
            st.markdown("---")

# Rodapé
st.markdown('<div class="footer">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
