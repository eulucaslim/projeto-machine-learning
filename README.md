
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
x