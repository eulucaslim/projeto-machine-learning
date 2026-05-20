import pandas as pd

print("Dataset do INMET com dados meteorológicos de Manaus\n")
# Carregamento de dados para um Dataframe
df = pd.read_csv('dados.csv', sep=';')
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

