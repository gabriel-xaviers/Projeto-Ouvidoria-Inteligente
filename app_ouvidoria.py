
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import carregar_dados, dataframe_manifestacoes, embeddings_sentence_transformers, destacar_score

st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")

MODELOS = {
    "Multilíngue MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BGE Small PT": "BAAI/bge-small-pt-v1.5",
}

@st.cache_data
def carregar_df():
    return dataframe_manifestacoes()

@st.cache_resource
def carregar_modelo(nome):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODELOS[nome])

@st.cache_data
def calcular_embeddings(textos, modelo_nome):
    modelo = carregar_modelo(modelo_nome)
    return modelo.encode(list(textos), normalize_embeddings=True, show_progress_bar=False)

df = carregar_df()

st.title("🏛️ Ouvidoria Inteligente")
st.caption("Protótipo de triagem semântica de manifestações cidadãs")

with st.sidebar:
    st.header("⚙️ Configurações")
    modelo_nome = st.selectbox("Modelo de embedding", list(MODELOS.keys()))
    top_k = st.slider("Top-k", 1, 10, 5)
    st.info("Na primeira execução, o modelo escolhido pode ser baixado pela biblioteca sentence-transformers.")

embeddings = calcular_embeddings(df["texto"].tolist(), modelo_nome)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"
])

with tab1:
    st.subheader("Busca Semântica")
    query = st.text_area("Descreva o problema", placeholder="Ex.: a rua está cheia de buracos...")
    if query.strip():
        modelo = carregar_modelo(modelo_nome)
        q = modelo.encode([query], normalize_embeddings=True)
        scores = cosine_similarity(q, embeddings)[0]
        idx = np.argsort(scores)[::-1][:top_k]
        for rank, i in enumerate(idx, 1):
            score=float(scores[i])
            st.markdown(f"### {rank}. {destacar_score(score)} {df.iloc[i]['id']} — {score:.3f}")
            st.write(f"**Categoria:** {df.iloc[i]['categoria_oficial']}")
            st.write(df.iloc[i]["texto"])
            st.divider()
    else:
        st.write("Digite uma descrição para consultar as manifestações semanticamente.")

with tab2:
    st.subheader("Base Completa")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if st.button("Gerar matriz de similaridade"):
        sim=cosine_similarity(embeddings)
        fig=px.imshow(sim, x=df.id, y=df.id, aspect="auto", color_continuous_scale="Viridis",
                      title="Matriz de similaridade — embeddings")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Baixar matriz CSV", pd.DataFrame(sim,index=df.id,columns=df.id).to_csv().encode("utf-8"),
                           "matriz_similaridade.csv","text/csv")

with tab3:
    st.subheader("Espaço Vetorial")
    metodo=st.radio("Método de redução", ["PCA","t-SNE"], horizontal=True)
    if metodo=="PCA":
        coords=PCA(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        perplexity=min(10,max(2,len(df)-1))
        coords=TSNE(n_components=2, random_state=42, perplexity=perplexity, init="random").fit_transform(embeddings)
    plotdf=pd.DataFrame({"x":coords[:,0],"y":coords[:,1],"id":df.id,
                         "categoria":df.categoria_oficial,"texto":df.texto})
    fig=px.scatter(plotdf,x="x",y="y",color="categoria",hover_name="id",hover_data=["texto"],
                   title=f"Representações em 2D — {metodo}")
    st.plotly_chart(fig,use_container_width=True)
    st.info("A proximidade visual mostra relações semânticas capturadas pelo embedding; ela não garante que as categorias oficiais formem clusters perfeitamente separados.")

with tab4:
    st.subheader("🧩 Chunking de manifestação longa")
    texto=st.text_area("Cole uma manifestação longa", height=220)
    estrategia=st.selectbox("Estratégia", ["RecursiveCharacterTextSplitter"])
    col1,col2=st.columns(2)
    with col1: chunk_size=st.number_input("chunk_size",100,1000,300,50)
    with col2: chunk_overlap=st.number_input("chunk_overlap",0,500,50,10)
    if chunk_overlap >= chunk_size:
        st.error("chunk_overlap deve ser menor que chunk_size.")
    elif st.button("Gerar chunks") and texto.strip():
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,
                                                 separators=["\n\n","\n",". "," ",""])
        chunks=splitter.split_text(texto)
        st.success(f"{len(chunks)} chunks gerados.")
        modelo=carregar_modelo(modelo_nome)
        emb= modelo.encode(chunks,normalize_embeddings=True,show_progress_bar=False)
        for i,(chunk,e) in enumerate(zip(chunks,emb),1):
            with st.expander(f"Chunk {i} — {len(chunk)} caracteres"):
                st.write(chunk)
                st.caption("Embedding: "+np.array2string(e[:10], precision=4)+" ...")
        if len(chunks)>1:
            coords=PCA(n_components=2,random_state=42).fit_transform(emb)
            cdf=pd.DataFrame({"x":coords[:,0],"y":coords[:,1],
                              "chunk":[f"Chunk {i}" for i in range(1,len(chunks)+1)]})
            st.plotly_chart(px.scatter(cdf,x="x",y="y",text="chunk",title="Chunks no espaço 2D"),use_container_width=True)
