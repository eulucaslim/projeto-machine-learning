import pandas as pd

# Carregamento de dados para um Dataframe
df = pd.read_csv('dados.csv', sep=';')
print(df.describe)