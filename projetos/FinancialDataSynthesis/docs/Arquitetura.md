# Detalhes da Arquitetura ##
==============================
<div align="center">
    <img src="../Arquitetura_Blocos.png" alt="Arquitetura em Blocos" title="Arquitetura em Blocos" />
    <p><em>Figura 1: Arquitetura da Rede Neural em Blocos.</em></p>
</div>


1.**Input:**

A entrada são as séries temporais do preço:

$$ X_{1:N} = [x(1), x(2), ..., x(N)] $$

E as séries temporais dos features, que por simplicidade, consideramos apenas uma série temporal, dada por:

$$ F_{1:N} = [f(1), f(2), ..., f(N)] $$

Essas séries são agrupadas em um mesmo dataframe, dado por:

$$ D = [X_{1:N},F_{1:N}] $$ 

2.**Sequenciador das Séries Temporais:**

Para que os dados possam ser processados pelos blocos Transformers, geramos sequências de tamanho fixo. No nosso caso, observamos que as sequências de tamanho 24 (tam_seq = 24) geraram os melhores resultados. Portanto, as séries temporais são separadas em sequências. Por exemplo, sequências da série temporal de preços são geradas como:

$$ Sequências = [{x(1), x(2), ..., x(24)}] , [{x(2), x(3), ..., x(25)}], ..., [{x(N-24), x(N-23), ..., x(N-1)}] $$

Cada sequência possui um target, valor qual devemos predizer. Para o nosso caso, como cada sequência tem 24 preços, devemos predizer o 25º elemento (25º preço), logo os targets de cada sequência são dados por:

$$ Targets = [x(25)] , [x(26)], ..., [x(N)] $$

Por exemplo, o target da sequência $[{x(1), x(2), ..., x(24)}]$ é $x(25)$.

3. **Layer de Input:**

Representa a entrada da rede neural. No nosso exemplo, são sequencias com 24 elementos, para cada feature, além dos targets.

4. **Embedding Layer:**

A Embedding Layer é uma camda densa responsável por projetar as sequências de entrada em um espaço de dimensão superior. Isso permite que o modelo capture características mais complexas dos dados.

- Função:
  
   Transformar as sequências de entrada de dimensão (tam_seq, nº de features) para (tam_seq, model_dim).
  
- Valores Utilizados:
  
  - tam_seq = 24  (tamanho das sequências).
  - model_dim = 64 (dimensão interna usada nas representações do modelo).
  - nº de features = 7 (Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), Stochastic Oscillator, Commodity Channel Index, Volume, MACD histogram, Money Flow Index)
  
- Operação:
 
  Aplicação de uma camada densa sem função de ativação: Embeddings= Dense(𝑚𝑜𝑑𝑒𝑙_𝑑𝑖𝑚)(Sequências de Entrada)
  
  Embeddings=Dense(model_dim)(Sequências de Entrada)

  Resultado: um tensor de dimensão (tam_seq, model_dim).
