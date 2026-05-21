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