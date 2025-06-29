import os
import joblib

# CORREÇÃO: Voltamos ao nome original do modelo, que foi o que enviámos.
MODEL_PATH = 'modelo_conciliacao.pkl'

def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERRO] Modelo não encontrado em: {MODEL_PATH}")
        raise RuntimeError("Arquivo de modelo ausente. Impossível iniciar a API.")
    try:
        modelo = joblib.load(MODEL_PATH)
        print("[INFO] Modelo de conciliação carregado com sucesso!")
        return modelo
    except Exception as e:
        print(f"[ERRO] Falha ao carregar modelo: {e}")
        raise RuntimeError("Erro ao carregar o modelo treinado.")
