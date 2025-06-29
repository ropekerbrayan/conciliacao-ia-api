from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from thefuzz import fuzz
import os
from fastapi.middleware.cors import CORSMiddleware

# Importa a função do outro ficheiro
from carregador_modelo import carregar_modelo

model = None  # Variável global para armazenar o modelo

# Inicializa o FastAPI
app = FastAPI(
    title="API de Conciliação Inteligente",
    description="Uma API que utiliza um modelo de Machine Learning para prever a probabilidade de uma cobrança e uma transação bancária serem correspondentes.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Entrada(BaseModel):
    cliente_cob: str
    info_pagador: str
    valor_cob: float
    valor_trans: float
    tipo_cob: str
    tipo_trans: str

def pre_processar_entrada(dados: Entrada):
    df = pd.DataFrame([dados.dict()])
    df['dif_valor'] = abs(df['valor_cob'] - df['valor_trans'])
    df['similaridade_nome'] = df.apply(lambda row: fuzz.token_sort_ratio(row['cliente_cob'], row['info_pagador']), axis=1)

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
    global model
    # --- CÓDIGO CORRIGIDO ---
    model = carregar_modelo()
    if model is None:
        raise RuntimeError("O modelo não pôde ser carregado. A API não pode iniciar.")

@app.get("/")
def home():
    return {"mensagem": "API de IA para conciliação pronta para receber previsões no endpoint /prever!"}

@app.post("/prever")
def prever(dados: Entrada):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não está carregado. A API pode estar a inicializar.")
    
    df_processado = pre_processar_entrada(dados)
    features_modelo = ['valor_cob', 'valor_trans', 'dif_valor', 'similaridade_nome', 'tipo_compativel']
    X_para_prever = df_processado[features_modelo]
    predicao_array = model.predict(X_para_prever)
    probabilidade_array = model.predict_proba(X_para_prever)
    predicao = bool(predicao_array[0])
    confianca = round(probabilidade_array[0][1], 4)
    return {"match": predicao, "confianca": confianca}
