# Ouvidoria Inteligente — Projeto completo

Projeto baseado no desafio de NLP fornecido, com comparação BoW/TF-IDF/embeddings, detecção de duplicatas, chunking e aplicativo Streamlit.

## 1. Estrutura

```text
Ouvidoria_Inteligente/
├── app_ouvidoria.py
├── utils.py
├── requirements.txt
├── README.md
├── data/
│   └── manifestacoes.json
├── notebooks/
│   ├── analise_comparativa.ipynb
│   ├── deteccao_duplicatas.ipynb
│   └── chunking_manifestacoes.ipynb
└── RELATORIO.pdf
```

## 2. Observação sobre o dataset

O enunciado original informa que existe um `manifestacoes.json` com 40 manifestações. Como o arquivo oficial não foi anexado junto do enunciado, este pacote inclui um **dataset demonstrativo com 40 registros**, criado para manter o projeto executável.

Se seu professor fornecer o `manifestacoes.json` oficial, substitua `data/manifestacoes.json` pelo arquivo oficial. Não é necessário alterar o código.

## 3. Executar no VS Code

Abra a pasta `Ouvidoria_Inteligente` no VS Code.

No terminal:

### Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_ouvidoria.py
```

Se `python` não funcionar:
```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py -m streamlit run app_ouvidoria.py
```

O Streamlit mostrará um endereço local, normalmente:
`http://localhost:8501`

## 4. Executar os notebooks

Com o ambiente virtual ativado:

```powershell
python -m pip install -r requirements.txt
jupyter notebook
```

Abra a pasta `notebooks` e execute as células na ordem.

Também é possível usar o VS Code com a extensão Jupyter.

## 5. Modelos de embedding

O app oferece:
- `paraphrase-multilingual-MiniLM-L12-v2`
- `BAAI/bge-small-pt-v1.5`

Na primeira utilização, a biblioteca pode baixar o modelo da internet. Depois, ele fica disponível no cache local.

## 6. Funcionalidades do app

### Busca Semântica
Consulta em linguagem natural e retorna as manifestações mais próximas por similaridade de cosseno.

### Base Completa
Exibe os 40 registros e permite gerar a matriz de similaridade.

### Espaço Vetorial
Permite visualizar os embeddings em 2D usando PCA ou t-SNE, com cores por categoria oficial.

### Chunking
Recebe um texto longo, permite configurar `chunk_size` e `chunk_overlap`, mostra os chunks e uma visualização dos seus embeddings.

## 7. Sobre os scores

A busca usa:
- 🟢 score > 0,7
- 🟡 score > 0,5
- 🔴 demais scores

## 8. Entregas

Os três notebooks correspondem às Entregas 1, 2 e 3. O `app_ouvidoria.py` corresponde à Entrega 4. O `RELATORIO.pdf` sintetiza as decisões e limitações do protótipo.

