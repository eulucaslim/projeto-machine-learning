from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
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

# Acumulado de chuva nas últimas 6h e 24h anteriores ao momento atual
# Isso dá ao modelo contexto sobre o estado úmido da atmosfera
df_filtrado['precip_acum_6h']  = df_filtrado['PRECIPITACAO TOTAL, HORARIO(mm)'].rolling(6).sum()
df_filtrado['precip_acum_24h'] = df_filtrado['PRECIPITACAO TOTAL, HORARIO(mm)'].rolling(24).sum()

# Componentes temporais — sazonalidade intra-diária e mensal
df_filtrado['hora'] = df_filtrado.index.hour
df_filtrado['mes']  = df_filtrado.index.month
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
    .gt(0.1)
    .astype(int)
)
df_filtrado = df_filtrado.dropna(subset=['precip_acum_6h', 'precip_acum_24h'])

print("Distribuição do target:")
print(df_filtrado['target_chuva_d+1'].value_counts())
print(f"% de horas com chuva: {df_filtrado['target_chuva_d+1'].mean()*100:.1f}%\n")


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
    'precip_acum_6h',
    'precip_acum_24h',
    'hora',
    'mes',
]

# Dados de Target e de features para treino e para teste
X_treino = treino[features]
y_treino = treino[target]

X_teste = teste[features]
y_teste = teste[target]

# Parametros utilizados
# max_depth=6  → limita a profundidade da árvore para evitar overfitting.
#                Sem limite, a árvore memoriza o treino e vai mal no teste.
# min_samples_split=50 → um nó só é dividido se tiver ao menos 50 amostras,
#                        evitando divisões em grupos minúsculos sem significado.
# min_samples_leaf=20  → cada folha precisa ter ao menos 20 amostras,
#                        garantindo que as previsões finais sejam generalizáveis.
# random_state=42      → garante reprodutibilidade dos resultados.
# class_weight='balanced' → corrige o desbalanceamento automaticamente.
# O sklearn pesa cada classe inversamente à sua frequência,
# fazendo o modelo "levar a sério" as horas com chuva mesmo sendo minoria.

modelo = DecisionTreeClassifier(
    # Define a profundidade máxima da árvore
    max_depth=6, 
    # Mínimo de amostras necessário para dividir um nó.
    min_samples_split=50,
    # Minimo de valor final das folhas
    min_samples_leaf=20,
    # Usado quando as classes estão desbalanceadas
    class_weight='balanced',
    # Define a semente aleatória para nao gerar diversas arvores
    random_state=42
)

modelo.fit(X_treino, y_treino)

y_pred = modelo.predict(X_teste)
y_pred_prob = modelo.predict_proba(X_teste)[:, 1]  # probabilidade da classe 1

print("=" * 60)
print("MÉTRICAS DE AVALIAÇÃO")
print("=" * 60)
# Precision, Recall e F1 separados por classe
print(classification_report(y_teste, y_pred, target_names=["Sem chuva (0)", "Com chuva (1)"]))

auc = roc_auc_score(y_teste, y_pred_prob)
print(f"AUC-ROC: {auc:.4f}\n")

# Retornando as features mais importantes
print("=" * 60)
print("IMPORTÂNCIA DAS VARIÁVEIS (ordem decrescente)")
print("=" * 60)
importancias = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
print(importancias)
print()

# Gerando a matriz de confusao com as importantes features.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1: Matriz de Confusão
cm = confusion_matrix(y_teste, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Sem chuva", "Com chuva"]).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Matriz de Confusão")

# Gráfico 2: Importância das features
importancias.plot(kind='barh', ax=axes[1])
axes[1].set_title("Importância das Variáveis")
axes[1].set_xlabel("Importância")
axes[1].invert_yaxis()

# Gráfico 3: Curva ROC
fpr, tpr, _ = roc_curve(y_teste, y_pred_prob)
axes[2].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
axes[2].plot([0,1], [0,1], 'r--', label="Aleatório")
axes[2].set_xlabel("Taxa de Falso Positivo")
axes[2].set_ylabel("Taxa de Verdadeiro Positivo")
axes[2].set_title("Curva ROC")
axes[2].legend()

plt.tight_layout()
plt.show()

# Gráfico 3: Primeiros níveis da árvore (interpretabilidade)
plt.figure(figsize=(20, 8))
plot_tree(
    modelo,
    feature_names=features,
    filled=True,
    max_depth=3,   # mostra só os 3 primeiros níveis para não poluir
    fontsize=9
)
plt.title("Estrutura da Árvore de Decisão (primeiros 3 níveis)")
plt.tight_layout()
plt.show()

# Versão texto da árvore (primeiros 3 níveis) — útil para o relatório
print(export_text(modelo, feature_names=features, max_depth=3))
