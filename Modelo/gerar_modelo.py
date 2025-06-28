import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from thefuzz import fuzz
import os

def gerar_dados_sinteticos(n_samples=2000):
    """Gera um DataFrame de dados sintéticos para conciliação."""
    print("Gerando dados sintéticos...")
    
    # Listas de base para gerar dados
    clientes_base = ['Tech Solutions', 'Inova Corp', 'Consultoria XYZ', 'Mercado Central', 'Loja Tech', 'Alpha Digital', 'Beta Services', 'Oficina Mecânica Águia']
    tipos_cobranca_base = ['Boleto', 'PIX QR Code', 'Credito 1x Visa', 'Credito 3x Master', 'Debito Elo']
    tipos_transacao_base = ['PIX', 'TED', 'CIELO', 'REDE', 'GETNET']

    data = []

    # --- Gerar exemplos POSITIVOS (match = 1) ---
    for i in range(n_samples // 2):
        cliente = np.random.choice(clientes_base)
        valor_cob = round(np.random.uniform(50.0, 5000.0), 2)
        
        # Variação 1: Match perfeito ou quase perfeito
        if np.random.rand() > 0.3:
            info_pagador_var = np.random.choice(["", " LTDA", " ME", " S.A.", " EIRELI"])
            info_pagador = cliente + info_pagador_var
            valor_trans = valor_cob
            tipo_cob = np.random.choice(tipos_cobranca_base)
            tipo_trans = "PIX" if "PIX" in tipo_cob else "TED"
        # Variação 2: Match com taxa de cartão
        else:
            info_pagador = cliente
            taxa = np.random.uniform(0.015, 0.05) # Taxa entre 1.5% e 5%
            valor_trans = round(valor_cob * (1 - taxa), 2)
            tipo_cob = np.random.choice(['Credito 1x Visa', 'Credito 3x Master', 'Debito Elo'])
            tipo_trans = np.random.choice(['CIELO', 'REDE', 'GETNET'])
        
        data.append([cliente, info_pagador, valor_cob, valor_trans, tipo_cob, tipo_trans, 1])

    # --- Gerar exemplos NEGATIVOS (match = 0) ---
    for i in range(n_samples // 2):
        cliente_cob = np.random.choice(clientes_base)
        info_pagador = np.random.choice(clientes_base)
        # Garante que os nomes sejam diferentes na maioria dos casos negativos
        while cliente_cob == info_pagador:
            info_pagador = np.random.choice(clientes_base) + " Pagamentos"

        valor_cob = round(np.random.uniform(50.0, 5000.0), 2)
        valor_trans = round(np.random.uniform(50.0, 5000.0), 2)
        # Garante que os valores sejam diferentes na maioria dos casos negativos
        if np.random.rand() > 0.1:
            while abs(valor_cob - valor_trans) < 1.0:
                 valor_trans = round(np.random.uniform(50.0, 5000.0), 2)
        
        tipo_cob = np.random.choice(tipos_cobranca_base)
        tipo_trans = np.random.choice(tipos_transacao_base)
        
        data.append([cliente_cob, info_pagador, valor_cob, valor_trans, tipo_cob, tipo_trans, 0])

    df = pd.DataFrame(data, columns=['cliente_cob', 'info_pagador', 'valor_cob', 'valor_trans', 'tipo_cob', 'tipo_trans', 'match'])
    return df.sample(frac=1).reset_index(drop=True)

def pre_processar_e_gerar_features(df):
    """Cria features numéricas para o modelo a partir dos dados brutos."""
    print("Pré-processando e criando novas features...")
    
    # 1. Diferença de valor (feature já existente na sua API)
    df['dif_valor'] = abs(df['valor_cob'] - df['valor_trans'])
    
    # 2. Similaridade de nomes (usando a biblioteca thefuzz)
    # Token Sort Ratio lida bem com palavras fora de ordem (ex: "Tech Solutions" vs "Solutions Tech LTDA")
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

def treinar_modelo():
    """Função principal que orquestra a criação e treinamento do modelo."""
    
    # Gera e processa os dados
    dados_sinteticos = gerar_dados_sinteticos(n_samples=5000)
    # >>>>>>>> A CORREÇÃO ESTÁ AQUI <<<<<<<<<<
    df_processado = pre_processar_e_gerar_features(dados_sinteticos)
    
    # Define as features que o modelo usará para aprender
    features = [
        'valor_cob',
        'valor_trans',
        'dif_valor',
        'similaridade_nome',
        'tipo_compativel'
    ]
    
    X = df_processado[features]
    y = df_processado['match']
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print(f"\nIniciando o treinamento do modelo com {len(X_train)} exemplos...")
    
    # Cria e treina o modelo RandomForest
    # É um bom modelo para começar, pois é robusto e lida bem com diferentes escalas de features
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    modelo.fit(X_train, y_train)
    
    print("Treinamento concluído!")
    
    # Avalia o modelo
    print("\nAvaliando o desempenho do modelo no conjunto de teste...")
    predicoes = modelo.predict(X_test)
    
    print("\nAcurácia:", accuracy_score(y_test, predicoes))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, predicoes))
    
    # Garante que o diretório de destino exista
    # O script é executado a partir de 'conciliacao-ia', então o caminho para a pasta 'modelo' é direto.
    output_dir = 'modelo'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Salva o modelo treinado
    nome_arquivo_modelo = os.path.join(output_dir, 'modelo_conciliacao.pkl')
    joblib.dump(modelo, nome_arquivo_modelo)
    
    print(f"\n✅ Modelo salvo com sucesso como '{nome_arquivo_modelo}'!")
    print("Agora você pode executar a sua API FastAPI.")

if __name__ == '__main__':
    treinar_modelo()