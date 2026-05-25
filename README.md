
# Projeto de IA - Pipeline de Machine Learning - Previsão de Chuva do dia 20 de maio
## 1. Aquisição

### 1.1. O que os dados representam e qual é o contexto do conjunto de dados?
Foi levantado dados meteorológicos da cidade de Manaus pelo site oficial do Instituto Nacional de Meteorologia (INMET) com o intuito de realizar a previsão de chuva do dia seguinte por meio desses dados, ou seja, lá tem todos os dados climáticos necessários para realizar essa análise, com as seguintes informações específicas:

Nome: MANAUS
Codigo Estacao: A101
Latitude: -3.10333333
Longitude: -60.01638888
Altitude: 61.25
Situacao: Operante
Data Inicial: 2026-05-01
Data Final: 2026-05-19
Periodicidade da Medicao: Horaria

### 1.2. Quantas linhas (amostras) e colunas (variáveis) há na base
Ao todo são 456 linhas (amostras) e 22 colunas em nosso dataset

### 1.3. Quais colunas existem e o que cada uma significa
- Data Medicao
- Hora Medicao
- PRECIPITACAO TOTAL, HORARIO(mm)
- PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(mB)
- PRESSAO ATMOSFERICA REDUZIDA NIVEL DO MAR, AUT(mB)
- PRESSAO ATMOSFERICA MAX.NA HORA ANT. (AUT)(mB)
- PRESSAO ATMOSFERICA MIN. NA HORA ANT. (AUT)(mB)
- RADIACAO GLOBAL(Kj/m²)
- TEMPERATURA DA CPU DA ESTACAO(°C)
- TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)
- TEMPERATURA DO PONTO DE ORVALHO(°C)
- TEMPERATURA MAXIMA NA HORA ANT. (AUT)(°C)
- TEMPERATURA MINIMA NA HORA ANT. (AUT)(°C)
- TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT)(°C)
- TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT)(°C)
- TENSAO DA BATERIA DA ESTACAO(V)
- UMIDADE REL. MAX. NA HORA ANT. (AUT)(%)
- UMIDADE REL. MIN. NA HORA ANT. (AUT)(%)
- UMIDADE RELATIVA DO AR, HORARIA(%)
- VENTO, DIRECAO HORARIA (gr)(° (gr))
- VENTO, RAJADA MAXIMA(m/s)
- VENTO, VELOCIDADE HORARIA(m/s)

### 1.4. Qual é a variável alvo (o que o modelo vai prever)
A ideia principal é determinar se irá chover ou não no dia 20 de maio de 2026 com base nos dados do dia 1 até o dia 19 desse mês.

## 2. Pré-processamento
### 2.1. Verificar e tratar valores ausentes (NaN / nulos)
Dependendo do valor que foi verificado nas Series, foi realizado um tratamento especifico, para dados da coluna de precipitacao foi adicionado 0 as valores null ou NaN, ja em outras foram realizadas de outras maneiras.

### 2.2 Decidir entre remover as linhas, preencher com média/mediana ou outro criterio
Para os valores meteorologicos tiveram que realizar o seguinte tratamento, foi utilizado a tecnica de Interpolacao temporal, que fazia o seguinte,
adiciona o valor de acordo com o tempo entre a linha anterior e a linha posterior do valor null e adicionava o valor referente. Alem disso, teve que ser feito a criacao de um indice em datetime para que essa conversao tivesse dado certo.

### 2.3 Identificar e tratar outliers, se houver
Nesse caso, para verificar a presenca de outliers, foi utilizado um metodo de verificacao chamado IQR (Interquartile Range) ou Intervalo Interquartil que consiste na seguinte maneira, possui um per-quartil - Q1 com valores de 25% e o Q3 com valores de 75%, e entao fazemos uma subtracao do Q3 - Q1 para descobrir o valor do IQR e entao verificamos o valor minimo e o maximo para nao ser considerado um outlier. 

### 2.4 Converter variáveis categóricas em número, se necessário
A maioria dos valores ja estava em float64 e entao nao foi preciso. 

### 2.5 Dividir os dados em conjuntos de treino e teste
Para isso a divisao foi feita com base em 80% dos dados serem de testes que seriam equivalentes a 14 dias e para teste seria os 20% dos dados do dia 15 ate dia 19.

# 3. Treinamento do Modelo
## 3.1 Como o modelo escolhido funciona, de forma intuitiva

O modelo escolhido foi a **Árvore de Decisão** (`DecisionTreeClassifier`), utilizada para classificação binária: prever se vai ou não vai chover nas próximas 24 horas.

De forma intuitiva, uma Árvore de Decisão funciona como um **fluxograma de perguntas encadeadas**. A cada etapa, o modelo faz uma pergunta sobre uma variável meteorológica e divide os dados em dois grupos — os que satisfazem a condição e os que não satisfazem. Esse processo se repete até que o modelo chegue a uma conclusão: **"vai chover"** ou **"não vai chover"**.

No caso deste projeto, a primeira pergunta que o modelo aprendeu foi:

> *"A radiação global é menor ou igual a 1017.6 Kj/m²?"*

Se a radiação for **maior** que esse valor — indicando um dia quente e ensolarado —, o modelo já classifica diretamente como **chuva**, capturando o padrão de chuvas convectivas da tarde, típico do clima de Manaus. Caso contrário, o modelo continua fazendo perguntas sobre temperatura do ponto de orvalho, pressão atmosférica e precipitação acumulada nas últimas 6 horas, até chegar a uma classificação final.

Essa estrutura é vantajosa para este trabalho porque:
- É **visualmente interpretável** — é possível desenhar e explicar cada decisão do modelo
- Captura **relações não-lineares** entre as variáveis, ao contrário da Regressão Linear
- Permite identificar **quais variáveis meteorológicas mais influenciam** a previsão de chuva

---

## 3.2 Quais parâmetros foram configurados e por quê

```python
modelo = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)
```

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `max_depth` | `6` | Limita a profundidade máxima da árvore. Sem esse limite, a árvore cresce até memorizar completamente os dados de treino (*overfitting*), perdendo a capacidade de generalizar para dados novos. A profundidade 6 equilibra complexidade e generalização. |
| `min_samples_split` | `50` | Define o número mínimo de amostras necessárias para que um nó seja dividido. Impede que o modelo crie divisões baseadas em grupos muito pequenos, que seriam estatisticamente irrelevantes. |
| `min_samples_leaf` | `20` | Garante que cada folha da árvore — ou seja, cada previsão final — seja baseada em pelo menos 20 casos históricos, tornando as previsões mais confiáveis. |
| `class_weight` | `'balanced'` | Corrige o desbalanceamento do dataset, onde apenas 11.8% das horas registram chuva. Sem esse parâmetro, o modelo tenderia a prever "sem chuva" para quase todos os casos, pois isso minimiza o erro médio. Com `'balanced'`, o sklearn atribui automaticamente um peso maior à classe minoritária (com chuva), forçando o modelo a levá-la a sério durante o treinamento. |
| `random_state` | `42` | Garante que os resultados sejam reproduzíveis. Como a Árvore de Decisão envolve escolhas que podem variar entre execuções, fixar essa semente garante que o mesmo resultado seja obtido toda vez que o código for executado. |

---

## 3.3 Como o modelo foi treinado

O treinamento foi realizado em três etapas principais:

**1. Separação temporal dos dados**

Os dados foram divididos respeitando a ordem cronológica — 80% para treino e 20% para teste — sem embaralhamento. Isso é essencial em séries temporais: embaralhar os dados vazaria informações do futuro para o treino, inflando artificialmente as métricas.

```python
valor_treino = int(len(df_filtrado) * 0.8)
treino = df_filtrado.iloc[:valor_treino]
teste  = df_filtrado.iloc[valor_treino:]
```

**2. Definição de features e target**

O target binário foi criado verificando se a precipitação nas próximas 24 horas supera 0.1mm (limiar para ignorar ruído numérico da interpolação):

```python
df_filtrado['target_chuva_d+1'] = (
    df_filtrado['PRECIPITACAO TOTAL, HORARIO(mm)']
    .shift(-24)
    .gt(0.1)
    .astype(int)
)
```

As features incluem variáveis meteorológicas horárias (pressão, radiação, temperatura, umidade, vento) e features de engenharia criadas para capturar contexto temporal: precipitação acumulada nas últimas 6h e 24h, hora do dia e mês do ano.

**3. Chamada do treinamento**

O treinamento foi executado com o método `.fit()`, que recebe as features e o target do conjunto de treino e ajusta internamente os limiares de cada nó da árvore para minimizar a impureza de Gini — critério padrão que mede o quão misturadas estão as classes em cada divisão:

```python
modelo.fit(X_treino, y_treino)
```

Após o treinamento, as previsões foram geradas com `.predict()` para a classificação final e `.predict_proba()` para obter a probabilidade associada à classe "com chuva", usada no cálculo da Curva ROC:

```python
y_pred      = modelo.predict(X_teste)
y_pred_prob = modelo.predict_proba(X_teste)[:, 1]
```

# 4. Avaliação do Modelo

## 4.1 Métricas utilizadas

A avaliação foi realizada sobre o conjunto de teste (20% dos dados, respeitando a ordem cronológica), composto por 87 amostras — sendo 81 horas sem chuva e apenas 6 horas com chuva, refletindo o forte desbalanceamento presente no dataset original (11.8% de horas chuvosas).

As métricas calculadas foram:

```
               precision    recall  f1-score   support

Sem chuva (0)       0.95      0.72      0.82        81
Com chuva (1)       0.12      0.50      0.19         6

     accuracy                           0.70        87
    macro avg       0.53      0.61      0.50        87
 weighted avg       0.89      0.70      0.77        87

AUC-ROC: 0.6183
```

**Definição das métricas:**

- **Precision**: de todos os casos que o modelo previu como "com chuva", qual proporção realmente choveu
- **Recall**: de todos os casos em que realmente choveu, qual proporção o modelo conseguiu detectar
- **F1-Score**: média harmônica entre Precision e Recall — penaliza quando uma das duas é muito baixa
- **AUC-ROC**: mede a capacidade do modelo de separar as duas classes em diferentes limiares de decisão. Varia de 0.5 (aleatório) a 1.0 (perfeito)

---

## 4.2 O modelo teve bom desempenho? Como você justifica essa afirmação?

O desempenho foi **parcial**: satisfatório para a classe "sem chuva" e insatisfatório para a classe "com chuva".

A **accuracy de 70%** não é um indicador confiável aqui. Como 88% das horas do teste são sem chuva, um modelo que previsse "nunca vai chover" para todos os casos já alcançaria 88% de acurácia sem aprender nada. Por isso, a análise foi centrada no **F1-Score por classe**.

**Classe "Sem chuva" — bom desempenho (F1 = 0.82)**

O modelo é confiável quando prevê ausência de chuva. A Precision de 0.95 indica que, quando diz que não vai chover, acerta em 95% dos casos.

**Classe "Com chuva" — desempenho fraco (F1 = 0.19)**

Este é o resultado mais relevante para o problema. A Precision de 0.12 significa que, a cada 10 alertas de chuva emitidos pelo modelo, apenas 1.2 se confirmam — ou seja, há muitos falsos alarmes. O Recall de 0.50 indica que o modelo detecta apenas metade dos eventos de chuva reais, perdendo a outra metade completamente.

O **AUC-ROC de 0.618** confirma que o modelo consegue separar as classes melhor que o acaso (0.5), mas está distante de um desempenho robusto. A curva ROC apresenta um formato em degrau característico de conjuntos de teste com pouquíssimos positivos — cada ponto na curva representa um dos apenas 6 casos de chuva sendo reclassificado.

A principal limitação não é o algoritmo em si, mas o **tamanho reduzido do conjunto de teste**, que conta com apenas 6 eventos de chuva — volume insuficiente para avaliar qualquer modelo de precipitação com confiança estatística.

---

## 4.3 Em que situações o modelo erra mais?

A Matriz de Confusão revela o padrão de erros com clareza:

|  | Previsto: Sem chuva | Previsto: Com chuva |
|---|---|---|
| **Real: Sem chuva** | 58 ✅ | 23 ❌ |
| **Real: Com chuva** | 3 ❌ | 3 ✅ |

O modelo comete dois tipos de erro:

**Falsos Positivos (23 casos):** prevê chuva quando não vai chover. Este é o erro mais frequente e é consequência direta do parâmetro `class_weight='balanced'`, que aumentou o peso da classe minoritária para forçar o modelo a detectar mais eventos de chuva — mas acabou tornando-o excessivamente "otimista" quanto à chuva.

**Falsos Negativos (3 casos):** não detecta chuva quando ela de fato ocorre. Embora menos frequente numericamente, este é o erro mais crítico do ponto de vista prático — uma previsão de "sem chuva" que falha representa o maior risco em aplicações reais.

Analisando a estrutura da árvore, o modelo erra principalmente em situações **meteorologicamente ambíguas**: radiação baixa, temperatura do ponto de orvalho próxima ao limiar de 23.75°C e precipitação acumulada nas últimas 6 horas entre 0.1mm e 3.8mm — faixa em que as condições não são claramente chuvosas nem claramente secas.

---

## 4.4 O que poderia ser feito para melhorar o resultado?

**1. Ampliar o dataset**

A limitação mais impactante foi o tamanho do conjunto de teste, com apenas 6 eventos de chuva. Incorporar mais anos de dados históricos do INMET aumentaria o número de eventos de chuva disponíveis para treino e teste, tornando as métricas estatisticamente mais confiáveis.

**2. Ajustar o balanceamento de classes manualmente**

Em vez de `class_weight='balanced'`, seria possível testar pesos customizados para encontrar um equilíbrio melhor entre Precision e Recall:

```python
class_weight={0: 1, 1: 5}  # penaliza 5x mais os erros na classe de chuva
```

**3. Adicionar features de tendência temporal**

O modelo usa valores instantâneos das variáveis, mas a variação ao longo do tempo carrega informação importante. Adicionar a variação da pressão nas últimas 3 horas, por exemplo, capturaria a queda de pressão que precede eventos chuvosos:

```python
df_filtrado['pressao_delta_3h'] = df_filtrado[
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(mB)'
].diff(3)
```

**4. Testar modelos mais robustos**

A Árvore de Decisão simples foi uma escolha adequada para fins didáticos e de interpretabilidade. Para uma aplicação real, modelos como Random Forest ou XGBoost — que combinam centenas de árvores — tenderiam a produzir resultados significativamente melhores, especialmente em datasets desbalanceados.