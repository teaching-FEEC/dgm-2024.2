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

A Embedding Layer é uma camada densa responsável por projetar as sequências de entrada em um espaço de dimensão superior. Isso permite que o modelo capture características mais complexas dos dados.

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

5. **Positional Encoding:**

   Como o Transformer não possui mecanismos recorrentes ou convolucionais, é necessário adicionar informações de posição para que o modelo entenda a ordem sequencial dos dados.

- Função:

 Incorporar informações de posição aos embeddings para que a sequência temporal seja considerada pelo modelo.
 
- Operação:

Geração de uma matriz de codificação posicional usando funções seno e cosseno:

PosEnc=Função Positional Encoding(𝑡𝑎𝑚_𝑠𝑒𝑞,𝑚𝑜𝑑𝑒𝑙_𝑑𝑖𝑚)

Adição das codificações posicionais aos embeddings:

Embeddings Posicionais =Embeddings + PosEnc

6. **Blocos Transformers:**

Os blocos Transformer são o núcleo do modelo, permitindo que ele aprenda relações complexas dentro das sequências.

- Função:

Processar as sequências posicionais através de mecanismos de atenção e redes feed-forward para capturar dependências temporais.

- Valores Utilizados:
- 
  - num_layers = 2: Número de blocos Transformer empilhados.
  - num_heads = 8: Número de cabeças no mecanismo de atenção múltipla.
  - ff_dim = 128: Dimensão da rede feed-forward interna.
  - dropout = 0.2: Taxa de dropout aplicada para evitar overfitting.
     
Operações em Cada Bloco Transformer:

**Multi-Head Attention:**

- Função:
  
 Permite que o modelo preste atenção a diferentes posições na sequência simultaneamente.
 
- Operação:

Attention Output = MultiHeadAttention(𝑛𝑢𝑚_ℎ𝑒𝑎𝑑𝑠,key_dim=𝑚𝑜𝑑𝑒𝑙_𝑑𝑖𝑚)(Input,Input)

Attention Output=MultiHeadAttention(num_heads,key_dim=model_dim)(Input,Input)

Aplicação de dropout na saída de atenção.


**Conexão Residual e Normalização**:

- Função:

Facilitar o fluxo de gradientes e estabilizar o treinamento.

- Operação: 

Output1 = LayerNormalization(Input+Attention Output)
Output1=LayerNormalization(Input+Attention Output)

**Feed-Forward Network (FFN):**

- Função:

Processar as representações aprendidas para capturar padrões não lineares.

- Operações:
  
Primeira camada densa com ativação ReLU e regularização L2:

FFN Output = Dense(𝑓𝑓_𝑑𝑖𝑚,activation=′𝑟𝑒𝑙𝑢′,kernel_regularizer=𝐿2)(Output1)

FFN Output=Dense(ff_dim,activation= ′relu ′,kernel_regularizer=L2)(Output1)

Segunda camada densa que retorna à dimensão model_dim:

FFN Output=Dense(model_dim,kernel_regularizer=L2)(FFN Output)
Aplicação de dropout na saída da FFN.

Conexão Residual e Normalização (2º Vez):

- Operação:

Output2=LayerNormalization(Output1+FFN Output)

Iteração: O processo acima é repetido para cada bloco Transformer (num_layers vezes), atualizando o input a cada iteração.


