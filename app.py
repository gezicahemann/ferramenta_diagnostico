import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar modelo de linguagem do spaCy
nlp = spacy.load("pt_core_news_sm")

# Carregar a base de dados
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Função de pré-processamento com spaCy
def preprocessar_texto(texto):
    doc = nlp(str(texto).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Aplicar pré-processamento ao campo "trecho"
df["trecho_processado"] = df["trecho"].apply(preprocessar_texto)

# Verificar se há dados válidos após pré-processamento
if df["trecho_processado"].str.strip().eq("").all():
    st.error("A base de dados está vazia após o pré-processamento. Verifique se há textos válidos no campo 'trecho'.")
    st.stop()

# Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Função de busca
def buscar_normas(consulta, limite=5, limiar_similaridade=0.3):
    consulta_processada = preprocessar_texto(consulta)
    if not consulta_processada:
        return pd.DataFrame()
    
    vetor_consulta = vetorizador.transform([consulta_processada])
    similaridades = cosine_similarity(vetor_consulta, matriz_tfidf).flatten()
    
    indices_relevantes = similaridades.argsort()[::-1][:limite]
    resultados = df.iloc[indices_relevantes].copy()
    resultados["similaridade"] = similaridades[indices_relevantes]
    resultados = resultados[ resultados["similaridade"] >= limiar_similaridade ]
    
    return resultados[["manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas", "similaridade"]]

# === Interface ===
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

# Estilo escuro com ajustes visuais refinados
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
    label {
        color: #d0d0d0 !important;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Logo centralizada
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("logo_engenharia.png", width=90)

# Título e instrução
st.markdown("<h1 style='text-align: center;'>🧱 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>", unsafe_allow_html=True)

# Campo de entrada
entrada = st.text_input("Descreva o problema:")

# Buscar e exibir resultados
if entrada:
    resultados = buscar_normas(entrada)

    if resultados.empty:
        st.warning("Nenhum resultado encontrado com base na manifestação informada.")
    else:
        st.markdown("### 🔍 Resultados encontrados:")
        for _, linha in resultados.iterrows():
            st.markdown(f"""
            ---
            **Manifestação:** {linha['manifestacao']}  
            **Norma:** {linha['norma']} (Seção {linha['secao']})  
            **Trecho técnico:** {linha['trecho']}  
            **Recomendações de verificação:** {linha['recomendacoes']}  
            **Consultas técnicas relacionadas:** {linha['consultas_relacionadas']}  
            """, unsafe_allow_html=True)

# Rodapé
st.markdown("<div class='footer'>Desenvolvido por Gézica Hemann | Engenharia Civil</div>", unsafe_allow_html=True)
