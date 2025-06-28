from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from thefuzz import fuzz
import os
# Importação para a correção de CORS
from fastapi.middleware.cors import CORSMiddleware

# Caminho para o modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'modelo', 'modelo_conciliacao.pkl')

# Tenta carregar o modelo treinado.
try:
    model = joblib.load(MODEL_PATH)
    print("Modelo de conciliação carregado com sucesso!")
except FileNotFoundError:
    print(f"Erro: Arquivo de modelo não encontrado em: {MODEL_PATH}")
    print("Por favor, execute o script 'modelo/gerar_modelo.py' primeiro para treinar e salvar o modelo.")
    model = None

# Inicializa o FastAPI
app = FastAPI(
    title="API de Conciliação Inteligente",
    description="Uma API que utiliza um modelo de Machine Learning para prever a probabilidade de uma cobrança e uma transação bancária serem correspondentes.",
    version="1.0.0"
)

# --- INÍCIO DA CORREÇÃO DE CORS ---
# Adiciona o "crachá de permissão" (Middleware)
# Isto permite que o seu dashboard (executado a partir de qualquer lugar)
# possa comunicar com a API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens (ex: file://, http://localhost)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os cabeçalhos
)
# --- FIM DA CORREÇÃO DE CORS ---


# Modelo de dados para a entrada da API
class Entrada(BaseModel):
    cliente_cob: str
    info_pagador: str
    valor_cob: float
    valor_trans: float
    tipo_cob: str
    tipo_trans: str

def pre_processar_entrada(dados: Entrada):
    """
    Recebe os dados da API e os transforma nas features que o modelo espera.
    Esta função deve ser IDÊNTICA à lógica de pré-processamento do script de treino.
    """
    df = pd.DataFrame([dados.dict()])

    # 1. Diferença de valor
    df['dif_valor'] = abs(df['valor_cob'] - df['valor_trans'])
    
    # 2. Similaridade de nomes
    df['similaridade_nome'] = df.apply(lambda row: fuzz.token_sort_ratio(row['cliente_cob'], row['info_pagador']), axis=1)

    # 3. Análise do tipo de pagamento
    def check_tipo_match(row):
        cob_lower = str(row['tipo_cob']).lower()
        trans_lower = str(row['tipo_trans']).lower()
        if 'pix' in cob_lower and 'pix' in trans_lower:
            return 1
        if 'boleto' in cob_lower and ('ted' in trans_lower or 'pix' in trans_lower):
            return 1
        if ('credito' in cob_lower or 'debito' in cob_lower) and trans_lower in ['cielo', 'rede', 'getnet', 'stone']:
            return 1
        return 0
    df['tipo_compativel'] = df.apply(check_tipo_match, axis=1)

    return df


@app.on_event("startup")
async def startup_event():
    if model is None:
        raise RuntimeError("O modelo não pôde ser carregado. A API não pode iniciar.")

@app.get("/")
def home():
    return {"mensagem": "API de IA para conciliação pronta para receber previsões no endpoint /prever!"}

@app.post("/prever")
def prever(dados: Entrada):
    """
    Recebe dados de uma cobrança e uma transação e retorna a predição do modelo.
    """
    # Pré-processa os dados recebidos
    df_processado = pre_processar_entrada(dados)

    # Define as features que o modelo precisa, na ordem correta
    features_modelo = [
        'valor_cob',
        'valor_trans',
        'dif_valor',
        'similaridade_nome',
        'tipo_compativel'
    ]
    
    # Garante que as colunas estejam na mesma ordem do treinamento
    X_para_prever = df_processado[features_modelo]

    # Realiza a predição
    predicao_array = model.predict(X_para_prever)
    probabilidade_array = model.predict_proba(X_para_prever)

    # Extrai os resultados
    predicao = bool(predicao_array[0])
    confianca = round(probabilidade_array[0][1], 4) # Retorna a probabilidade da classe "1" (match)

    return {
        "match": predicao,
        "confianca": confianca
    }
