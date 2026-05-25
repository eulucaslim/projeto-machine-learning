from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error, 
    r2_score
)
from utils import get_outliers
import matplotlib.pyplot as plt
import pandas as pd

print("Dataset do INMET com dados meteorológicos de Manaus\n")
# Carregamento de dados para um Dataframe
df = pd.read_csv('dados.csv', sep=';', decimal=',')
# Exibe o começo do Dataframe para uma listagem simples dos dados
print(df.head)

# Remove as colunas listadas abaixo com seus respectivos motivos
df_filtrado = df.drop(
    columns=[
        # Por se tratar de dados operacionais
        'TEMPERATURA DA CPU DA ESTACAO(°C)',  
        'TENSAO DA BATERIA DA ESTACAO(V)',

        # Informações duplicadas - redundância informacional
        'PRESSAO ATMOSFERICA MAX.NA HORA ANT. (AUT)(mB)',
        'PRESSAO ATMOSFERICA MIN. NA HORA ANT. (AUT)(mB)',
        'TEMPERATURA MAXIMA NA HORA ANT. (AUT)(°C)',
        'TEMPERATURA MINIMA NA HORA ANT. (AUT)(°C)',
        'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT)(°C)',
        'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT)(°C)',
        'UMIDADE REL. MAX. NA HORA ANT. (AUT)(%)',
        'UMIDADE REL. MIN. NA HORA ANT. (AUT)(%)'
    ]
)

print("=" * 60, "\n")
print("Essas foram as colunas restantes: \n")
for col in df_filtrado.columns:
    print(f"- {col}")

print("\nQuantidade de colunas restantes: ", len(df_filtrado.columns))
print("=" * 60, "\n")

# 2. Identificação de Valores nulos e NaN
print("Esses são os valores nulos e NaN encontrados em todas as colunas") 
print(df_filtrado.isna().sum(), "\n")
print("=" * 60, "\n")

# Percentual de ausência
# % de NaN / Ação
# < 5% / Imputar
# 5–20%	/ Avaliar
# > 30%	/ Considerar remover
print("Percentual de ausência \n", (df_filtrado.isna().sum() / len(df_filtrado)) * 100, "\n")

# Tratamento para os valores nulos, para precipitação
# Coluna Precipitação Total - Ausência significa zero precipitação
df_filtrado["PRECIPITACAO TOTAL, HORARIO(mm)"] = df_filtrado[
    "PRECIPITACAO TOTAL, HORARIO(mm)"].fillna(0)

# Colunas Temperatura, pressão, umidade, orvalho
# Forçar string o que era o Int dos horários
df_filtrado["Hora Medicao"] = df_filtrado["Hora Medicao"].astype(str).str.zfill(4)

# Criação do Indice pelo datetime 
df_filtrado["datetime"] = pd.to_datetime(
    df_filtrado["Data Medicao"] + " " + df_filtrado["Hora Medicao"],
    format="%Y-%m-%d %H%M"
)
df_filtrado = df_filtrado.set_index("datetime").sort_index()
df_filtrado = df_filtrado.asfreq("h")

print("Frequência do Índice: ", df_filtrado.index.freq)

cols_interpolar = [
    "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(mB)",
    "PRESSAO ATMOSFERICA REDUZIDA NIVEL DO MAR, AUT(mB)",
    "RADIACAO GLOBAL(Kj/m²)",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)",
    "TEMPERATURA DO PONTO DE ORVALHO(°C)",
    "VENTO, RAJADA MAXIMA(m/s)",
    "VENTO, VELOCIDADE HORARIA(m/s)",
    "PRECIPITACAO TOTAL, HORARIO(mm)",
    "UMIDADE RELATIVA DO AR, HORARIA(%)"
]

df_filtrado[cols_interpolar] = df_filtrado[cols_interpolar].interpolate(method="time")

# Colunas Vento (direção, velocidade, rajada)
df_filtrado["VENTO, DIRECAO HORARIA (gr)(° (gr))"] = (
    df_filtrado["VENTO, DIRECAO HORARIA (gr)(° (gr))"]
    .ffill()
)
# Coluna Pressao Atmosferica - Utiliza o dado futuro para explicar o passado
df_filtrado["PRESSAO ATMOSFERICA REDUZIDA NIVEL DO MAR, AUT(mB)"] = df_filtrado["PRESSAO ATMOSFERICA REDUZIDA NIVEL DO MAR, AUT(mB)"].bfill()
print("\nEsses foram a quantidade de valores nulos após o pré-processamento\n", df_filtrado.isna().sum(), '\n')
print("=" * 60, "\n")

# Verificacao de outliers em todas as colunas
outliers = pd.DataFrame()
for col in df_filtrado.columns:
    if df_filtrado[col].dtype == 'float64':
        outliers += get_outliers(df_filtrado, col) 

print("\nEsses foram os outliers detectados: ", outliers)
print("=" * 60, "\n")

# Criando o target para o dia seguinte - dado que queremos prever
df_filtrado['target_chuva_d+1'] = (
    df_filtrado['PRECIPITACAO TOTAL, HORARIO(mm)']
    .shift(-24)
)
df_filtrado = df_filtrado.dropna(subset=['target_chuva_d+1'])

# Dividindo os dados como treino (80% da quantidade total) e teste (20% da quantidade total)
valor_treino = int(len(df_filtrado) * 0.8)
treino = df_filtrado.iloc[:valor_treino]
teste = df_filtrado.iloc[valor_treino:]

# Mapeando colunas de Target e Features
target = 'target_chuva_d+1'
features = [
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(mB)',
    'PRESSAO ATMOSFERICA REDUZIDA NIVEL DO MAR, AUT(mB)',
    'RADIACAO GLOBAL(Kj/m²)',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)',
    'TEMPERATURA DO PONTO DE ORVALHO(°C)',
    'UMIDADE RELATIVA DO AR, HORARIA(%)',
    'VENTO, VELOCIDADE HORARIA(m/s)',
    'VENTO, RAJADA MAXIMA(m/s)',
]

# Dados de Target e de features para treino e para teste
X_treino = treino[features]
y_treino = treino[target]

X_teste = teste[features]
y_teste = teste[target]

modelo = LinearRegression(
    fit_intercept=True,  # A precipitação não começa em zero
    n_jobs=-1            # usa todos os núcleos da CPU
)

modelo.fit(X_treino, y_treino)

# Coeficientes 

print("Esses sao os valores que mais influenciam a chuva em ordem decrescente: \n")
coeficientes = pd.Series(
    modelo.coef_,
    index=X_treino.columns
).sort_values(ascending=False)

print(coeficientes)

# Gerando previsoes no conjunto de teste
y_pred = modelo.predict(X_teste)

mse = mean_squared_error(y_teste, y_pred) # erro médio interpretáve
rmse = root_mean_squared_error(y_teste, y_pred) # penaliza erros grandes (chuvas fortes)
mae = mean_absolute_error(y_teste, y_pred)
r2 = r2_score(y_teste, y_pred) # capacidade explicativa

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

plt.scatter(y_teste, y_pred, alpha=0.5)
plt.xlabel("Chuva Real (mm)")
plt.ylabel("Chuva Prevista (mm)")
plt.title("Real vs Previsto")
plt.show()