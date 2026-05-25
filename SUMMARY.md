# Possíveis perguntas 
## O que é uma feature em Machine Learning
```
uma definição bem simples e correta seria que uma feature é uma variável usada pelo modelo para aprender padrões e fazer as previsões.
```

## Feature x Target
```
A feature é a informação de entrada e o target é o que eu quero prever, nesse caso a feature são os dados de umidade, pressão,
temperatura e vento e o Target é se vai chover amanhã. 
```

## Por que foram removidas as colunas que possuiam as informação de Min/Max do Dataset
```
Pelo fato de se tratar da mesma informação passada anteriormente, ou seja, não são fenômenos novos e sim derivações diretas de váriaveis que já temos e isso causa um fênomeno chamado de Multicolinearidade.
``` 

## O que é interpolação
```
É o processo de estimular um valor faltante com base nos vizinhos mais próximos, ou seja, gera uma estimativa com base em um determinado comportamento de dados
```

## O que é interpolação linear
```
Traça uma linha reta entre dois pontos e pega o valor entre eles
```

## O que é interpolação temporal
```
Distância real no tempo por meio de timestamp
```

## Por que os dados de Precipitação Total (mm) foram tratados com 0
```
Em dados meteorológicos, ausência de precipitação registrada indica evento nulo, não ausência de medição.
```

## Por que a Pressão reduzida ao nível do mar foi utilizada a forma de interpolação para tratamento?
```
Para esse caso a pressão atmosféricas diminui lentamente e de forma contínua, então tornando a interpolação temporal mais viável para esse caso para pequenos intervalos ausentes.
```

## Por que foi necesssário a criação de um DatetimeIndex ?
```
Por conta que na tratativa de dados foi utilizado a técnica de interpolação temporal para esses dados e era necessário saber a distância entre os dados de forma que o pandas pudesse distinguir temporalmente, dessa forma foi necessário a criação da Series datetime para suprir essa necessidade.
```

## O que seriam outliers?
```
Sao os valores encontrados em nosso dataset que estao com uma diferenca muito grande de valor em relacao aos outros, e para trata-los existem varias maneiras por IQR, Z-Score e Quartil, a forma utilizada na nossa verificacao foi o IQR
```

## O que seria uma Arvore de Decisao
```
Uma Árvore de Decisão é um modelo de Inteligência Artificial que toma decisões passo a passo, de forma parecida com o raciocínio humano.
```