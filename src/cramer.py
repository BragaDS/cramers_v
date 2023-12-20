
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

# Função para calcular o coeficiente de Cramer's V entre duas colunas categóricas
def calculate_cramers_v(df, col1, col2):
    # Cria uma tabela de contingência entre as duas colunas
    contingency_table = pd.crosstab(df[col1], df[col2])
    
    # Calcula o teste qui-quadrado e outros valores associados
    chi2, _, _, _ = chi2_contingency(contingency_table)
    
    # Calcula o coeficiente de Cramer's V
    n = df.shape[0]
    v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    return v

# Função para identificar pares de variáveis altamente correlacionadas
def identify_highly_correlated_variables(df, target_column, threshold=0.9):
    # Conjunto para armazenar pares de colunas já processados
    processed_pairs = set()

    # Loop sobre todas as combinações de colunas no DataFrame, excluindo a coluna alvo
    for col1 in df.columns.drop(target_column):
        for col2 in df.columns.drop(target_column):
            # Verifica se as colunas são diferentes e ainda não foram processadas em ambas as direções
            if col1 != col2 and (col1, col2) not in processed_pairs and (col2, col1) not in processed_pairs:
                # Calcula o coeficiente de Cramer's V para o par de colunas
                cramer_v = calculate_cramers_v(df, col1, col2)
                
                # Se a correlação é maior que o limiar definido, imprime a informação
                if cramer_v > threshold:
                    print(f"As variáveis {col1} e {col2} têm o coeficiente de Cramer's V de {cramer_v}, indicando uma correlação significativa.")
                    
                    # Adiciona o par de colunas ao conjunto de pares processados
                    processed_pairs.add((col1, col2))

# Exemplo de uso da função para identificar variáveis altamente correlacionadas com a coluna alvo "Variavel_Alvo"                    
#identify_highly_correlated_variables(dataframe_utilizado, "Variavel_Alvo", Ajuste_do_threshold)