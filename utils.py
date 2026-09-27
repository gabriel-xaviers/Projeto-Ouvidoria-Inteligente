
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "manifestacoes.json"

def carregar_dados(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dataframe_manifestacoes(path=DATA_PATH):
    return pd.DataFrame(carregar_dados(path))

def normalizar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def matriz_bow(textos):
    vec = CountVectorizer(lowercase=True, strip_accents="unicode")
    X = vec.fit_transform([normalizar_texto(t) for t in textos])
    return X, vec

def matriz_tfidf(textos):
    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1,2))
    X = vec.fit_transform([normalizar_texto(t) for t in textos])
    return X, vec

def cosine_matrix(X):
    return cosine_similarity(X)

def pares_acima_limiar(sim, ids, limiar=0.85):
    pares=[]
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            if sim[i,j] >= limiar:
                pares.append((ids[i], ids[j], float(sim[i,j])))
    return sorted(pares,key=lambda x:x[2],reverse=True)

def embeddings_sentence_transformers(textos, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(list(textos), normalize_embeddings=True, show_progress_bar=False)

def detectar_duplicatas(textos, limiar=0.85, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    emb = embeddings_sentence_transformers(textos, model_name)
    sim = cosine_similarity(emb)
    pares=[]
    for i in range(len(textos)):
        for j in range(i+1,len(textos)):
            if sim[i,j] >= limiar:
                pares.append((i,j,float(sim[i,j])))
    return pares, sim

def buscar_semantica(query, df, embeddings, model, top_k=5):
    q = model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(q, embeddings)[0]
    idx = np.argsort(scores)[::-1][:top_k]
    out=df.iloc[idx].copy()
    out["score"]=scores[idx]
    return out

def destacar_score(score):
    if score > 0.7: return "🟢"
    if score > 0.5: return "🟡"
    return "🔴"
