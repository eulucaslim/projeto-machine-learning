
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

## 3. Treinamento do Modelo
### 3.1 Como o modelo escolhido funciona, de forma intuitiva
O modelo escolhido para resolver esse problema se chama Regressao Linear e funciona da seguinte maneira: Tenta encontrar uma linha (ou plano) que melhor se ajuste aos dados, minimizando os erros entre os valores reais e os valores previstos, aprendendo os pesos para cada feature, no nosso caso o modelo aprende quanto cada variável meteorológica influencia a chuva.

### 3.2 Quais parâmetros foram configurados e por quê
Foram adicionado os dois parametros o de 'fit_intercept=True' que aprende o termo independente, ou seja, a precipitacao nao comeca em zero e o 'n_jobs=-1' utiliza todos os nucleos da CPU para ter um melhor desempenho computacional.

### 3.3 Como o modelo foi treinado (qual funcao/metodo) foi chamado
A biblioteca utilizada para realizar o treinamento foi o sklearn chamando a classe do treinamento do modelo LinearRegression e para treinar utilizamos os dados de feature e de target chamando o metodo fit que calcula os coeficientes minimizando o erro quadratico (MSE) e tambem aprende a relacao entre o clima e a precitacao.