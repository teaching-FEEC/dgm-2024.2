# Detalhes da Arquitetura ##
==============================
<div align="center">
    <img src="../Arquitetura_Blocos.png" alt="Arquitetura em Blocos" title="Arquitetura em Blocos" />
    <p><em>Figura 1: Arquitetura da Rede Neural em Blocos.</em></p>
</div>


1.**Entrada (Input):**

As entradas do modelo são as séries temporais de preços e as séries temporais dos features.

A série temporal dos preços é dada por:

$$ X_{1:N} = [x(1), x(2), ..., x(N)] $$

As séries temporais de cada feature são representadas como:

$$ F_{1:N} = [f(1), f(2), ..., f(N)] $$

Essas séries são agrupadas em um mesmo dataframe, constituindo a entrada do modelo, dado por:

$$ D = [X_{1:N},F_{1:N}] $$ 

2.**Sequenciador das Séries Temporais:**

Para que os dados possam ser processados pelo modelo Transformer, geramos sequências de tamanho fixo a partir das séries temporais. No nosso caso, utilizamos sequências de tamanho tam_seq= 24, pois o modelo apresentou melhores resultados com este tamanho. 

As sequências dos preços são descritas como:

$$ Sequências = [{x(1), x(2), ..., x(24)}] , [{x(2), x(3), ..., x(25)}], ..., [{x(N-24), x(N-23), ..., x(N-1)}] $$

O mesmo procedimento é realizado para os features, ao final, juntamos as sequências de preço e features.

Cada sequência possui um target, valor qual devemos predizer. Para o nosso caso, como cada sequência tem 24 preços, devemos predizer o 25º elemento (25º preço), logo os targets de cada sequência são dados por:

$$ Targets = [x(25)] , [x(26)], ..., [x(N)] $$

Por exemplo, o target da sequência $[{x(1), x(2), ..., x(24)}]$ é $x(25)$.

3. **Input Layer:**

A camada de entrada da rede neural recebe as sequências de entrada. Cada sequência tem dimensão (tam_seq, n_features), em que:

 - tam_seq = 24  (tamanho das sequências).
 - n_features = 7 (número de features, sendo eles: Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), Stochastic Oscillator, Commodity Channel Index, Volume, MACD histogram, Money Flow Index)

4. **Embedding Layer:**

A Embedding Layer é uma camada densa responsável por projetar as sequências de entrada em um espaço de dimensão superior. Isso permite que o modelo capture características mais complexas dos dados.

- Função: transformar as sequências de entrada de dimensão (tam_seq, n_features) para (tam_seq, model_dim).
  
- Valor Utilizado:
  
  - model_dim = 64 (dimensão interna usada nas representações do modelo).
  
- Operação:
 
  Aplicação de uma camada densa sem função de ativação:
  
  Embeddings=Dense(model_dim)(Sequências de Entrada)

  Resultado: um tensor de dimensão (tam_seq, model_dim).

5. **Positional Encoding:**

   Como o Transformer não possui mecanismos recorrentes ou convolucionais, é necessário adicionar informações de posição para que o modelo entenda a ordem sequencial dos dados.

- Função: incorporar informações de posição aos embeddings para que a sequência temporal seja considerada pelo modelo.
 
- Operação:

Geração de uma matriz de codificação posicional usando funções seno e cosseno:

PosEnc=Função Positional Encoding(𝑡𝑎𝑚_𝑠𝑒𝑞,𝑚𝑜𝑑𝑒𝑙_𝑑𝑖𝑚)

Adição das codificações posicionais aos embeddings:

Embeddings Posicionais = Embeddings + PosEnc

6. **Blocos Transformers:**

Os blocos Transformer são o núcleo do modelo, permitindo que ele aprenda relações complexas dentro das sequências.

- Função: processar as sequências posicionais através de mecanismos de atenção e redes feed-forward para capturar dependências temporais.

- Valores Utilizados:
  
  - num_layers = 2: Número de blocos Transformer empilhados.
  - num_heads = 8: Número de cabeças no mecanismo de atenção múltipla.
  - ff_dim = 128: Dimensão da rede feed-forward interna.
  - dropout = 0.2: Taxa de dropout aplicada para evitar overfitting.
     
Operações em Cada Bloco Transformer:

**Multi-Head Attention:**

- Função: permite que o modelo preste atenção a diferentes posições na sequência simultaneamente.
 
- Operação:
  
Attention Output=MultiHeadAttention(num_heads,key_dim=model_dim)(Input,Input)

Aplicação de dropout na saída de atenção.


**Conexão Residual e Normalização**:

- Função: facilitar o fluxo de gradientes e estabilizar o treinamento.

- Operação: Adição da entrada do bloco Transformer (Input) à saída do Layer Multi-Head Attention (Attention Output) e Normalização.

Output1 = LayerNormalization(Input+Attention Output)

**Feed-Forward Network (FFN):**

- Função: processar as representações aprendidas para capturar padrões não lineares.

- Operações:
  
Primeira camada densa com ativação ReLU e regularização L2:

FFN Output=Dense(ff_dim,activation= ′relu ′,kernel_regularizer=L2)(Output1)

Segunda camada densa que retorna à dimensão model_dim:

FFN Output=Dense(model_dim,kernel_regularizer=L2)(FFN Output)

Aplicação de dropout na saída da FFN.

**Conexão Residual e Normalização (2º Vez):**

- Operação:

Output2=LayerNormalization(Output1+FFN Output)

**Iteração:** O processo é repetido para cada bloco Transformer (num_layers vezes), atualizando o input a cada iteração.

7. **Global Average Pooling:**

Após passar pelos blocos Transformer, reduzimos a dimensionalidade para preparar os dados para a camada de saída (output layer).

- Função: reduzir a dimensão sequencial calculando a média das representações ao longo do tempo.
  
- Operação:
  
Aplicação de média global na dimensão temporal: 

Pooled_Output=GlobalAveragePooling1D(Output2)

Resultado: um vetor de dimensão (model_dim). 

8. **Output Layer:**
   A camada de saída produz a previsão final do modelo.

- Função: mapear as representações aprendidas para a dimensão do target desejado.
  
- Valores Utilizados:
    - output_dim = 1: Dimensão da saída, correspondente à previsão do retorno.

- Operação:

Aplicação de uma camada densa sem função de ativação:

Predição = Dense(𝑜𝑢𝑡𝑝𝑢𝑡_𝑑𝑖𝑚)(Pooled_Output)

Resultado: um valor escalar que representa a previsão do retorno no próximo período.
