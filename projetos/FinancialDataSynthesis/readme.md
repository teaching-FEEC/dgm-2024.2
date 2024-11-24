# `Síntese de Dados Financeiros`
==============================

# `Financial Data Synthesis`
==============================
## Link dos slides:
https://docs.google.com/presentation/d/1eOmgRpkQeXU1htM_7Gq66HRcn2CPZ7iB/edit?pli=1#slide=id.p1

Apresentação
==============================
<p align="justify">
O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação IA376N - Deep Learning aplicado a Síntese de Sinais, oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).
</p>

 |Nome  | RA | Curso|
 |--|--|--|
 |José Carlos Ferreira  | 170860  | Eng. Elétrica |
 |Byron Alejandro Acuña Acurio  | 209428  | Eng. Elétrica |

## Resumo (Abstract)
<p align="justify">
Neste trabalho foi testada a capacidade de dois modelos generativos baseadas em GANS e Transformers na previsão do preço da ação da empresa Apple, considerando uma
condição de evento extremo neste caso o Covid19 no ano 2020. A Rede Adversarial Generativa (GAN) foi desenvolvida usando Unidades Recorrentes Fechadas (GRU) como um gerador que insere o preço histórico das ações e gera a previsão do preço futuro (no día seguente) das ações e uma Rede Neural Convolucional (CNN) como um discriminador para discriminar entre o preço real das ações e o preço das ações gerado. Para estimar o preço da ação foi usada 36 características como índice S&P 500, índice NASDAQ Composite, índice U.S. Índice do dólar, etc. Comparamos os resultados do nosso modelo GAN e Transformer com modelos de aprendizado profundo baseados em LSTM e GRU. O modelo generativo apresenta um melhor desempenho em eventos extremos.
</p>

## Introdução
<p align="justify">
Nosso projeto foca na geração de dados financeiros sintéticos realistas, especificamente sobre o preço da ação da empressa Apple através de duas abordagens: baseadas em GANS e Transformers.
Os dados sintéticos são úteis em modelos em que a confiança apenas em dados históricos não é suficiente para construir um método robusto. Neste trabalho os experimentos foram realizados antes e depois do Covid-19, para verificar a robustez de nossos modelos generativos frente a eventos extremos.
</p>

O projeto lida com séries temporais da forma:

$$ X_{1:N}  = [{ x(1), x(2), ..., x(N) }]  $$

Em que cada elemento $x(i)$ representa o preço da ação da empressa Apple no instante $i$.

<p align="justify">
A continuação se apresenta a serie temporal dos preços da ação da empressa Apple, a usada data usada dos dados foi desde 2010-07-01 até 2020-06-30, para fazer experimentos antes e depois do Covid-19 (evento extremo)
</p>

![Time Series Visualization](img_readme/Serie_temporal.png)


Atráves da incorporação de features relevantes, também representados por séries temporais (alinhadas à $X_{1:N}$), buscamos gerar dados sintéticos que representam uma continuação realista de $X_{1:N}$, isso é, uma série temporal do tipo:

$$ X^{s}_{N+1:N+K}  = [{ x^{s}(N+1), x^{s}(N+2), ..., x^{s}(N+K) }]  $$

Tal que:

$$ X^{s}_{N+1:N+K}  \approx X\_{N+1:N+K}   $$



Por exemplo, se $X_{1:N}$ representa os preços da ação da empressa Apple janeiro até fevereiro, $X^{s}_{N+1:N+K}$ poderia representar valores plausíveis de fevereiro até março no ano 2020 que iniciu o lockdown por causa do Covid 19 (Evento Extremo).

<!-- Essas representações realistas são importantes para modelos de otimização de portfólios, pois podemos gerar diversos cenários possíveis e escolher a estratégia que se sai melhor, considerando todas as possibilidades geradas. Dessa forma, o modelo de otimização é robusto e consegue bom desempenho nas mais diversas situações. -->
## Descrição do Problema/Motivação
No setor financeiro, o acesso a dados do mundo real para análise e treinamento de modelos é limitado devido a questões de privacidade e segurança. Assim os dados sintéticos podem ajudar a fornecer uma alternativa segura para disponibilizar esses dados para diversas organizações. O desenvolvimento de modelos com capacidade de prever o preço da ação de forma precisa é desafiador devido à complexidade inerente desses dados. Em geral, os dados financeiros são não estacionários e seguem distribuições de probabilidade desconhecidas e difíceis de serem estimadas. Apesar dos avanços nos algoritmos de deep learning, que conseguem capturar melhor essas complexidades, a escassez de dados financeiros disponíveis tem sido um fator limitante na construção de métodos robustos. Especialmente em eventos extremos quando no histórico de dados nunca se teve um registro de um evento similar.

Há um movimento crescente entre pesquisadores para otimizar modelos de machine learning através da incorporação de dados financeiros sintéticos [4]. A geração de dados sintéticos permite melhorar o desempenho de métodos que, até então, apresentavam resultados insatisfatórios ou eram inviáveis na prática devido à falta de dados, além de possibilitar a simulação de eventos raros ou extremos. 

Diversas metodologias têm sido estudadas. As arquiteturas da família Generative Adversarial Networks (GANs) têm mostrado bons resultados em tarefas de geração de imagens e, mais recentemente, estão sendo aplicadas na geração de dados financeiros sintéticos. Além das GANs, as arquiteturas Transformers também surgem como estruturas promissoras para a tarefa. 

A criação de dados financeiros que reproduzam o comportamento de dados reais é essencial para várias aplicações, como o problema de otimização de portfólios. Considere um investidor com acesso a 𝑛 classes de ativos. O problema de otimização de portfólio consiste em alocar esses ativos de modo a maximizar o retorno, escolhendo a quantidade apropriada para cada classe, enquanto mantém o risco do portfólio dentro de um nível de tolerância predefinido. Pesquisas recentes em otimização de portfólios financeiros exploraram diversas abordagens para melhorar as estratégias de alocação de ativos. A geração de dados sintéticos tem se destacado como uma boa solução para ampliar conjuntos de dados financeiros limitados, com estudos propondo modelos de regressão sintética [1] e redes adversárias generativas condicionais modificadas [2].

Neste trabalho, focamos na geração de dados sintéticos de ativos listados em bolsas de valores (nacionais e internacionais) utilizando uma abordagem baseada em GANs e Transformers. A geração de dados sintéticos é particularmente útil para capturar cenários de retorno que estão ausentes nos dados históricos, mas são estatisticamente plausíveis.


## Objetivos

O projeto tem como **objetivo principal** :

- Gerar séries temporais sintéticas realistas de ativos financeiros.

Para o projeto, escolhemos três ativos financeiros distintos:
- **Índice Bovespa**: pontuação que mede o desempenho das ações das maiores empresas listadas na bolsa de ações brasileira (B3);
- **Índice S&P 500**: pontuação que mede o desempenho das 500 maiores ações listadas na bolsa de ações de Nova York (NYSE);
- **Ações da VALE S.A**: terceira maior empresa brasileira, com ações negociadas na NYSE e B3;

Além disso, adotamos duas abordagens distintas para geração dos dados:
1. Baseada na arquitetura **Transformers**;
2. Baseada na arquitetura de redes generativas adversarias **(GANs)**;

Temos como missão, dado a série temporal desses ativos em determinado período, gerar séries temporais sintéticas plausíveis que representam a continuação das séries originais.

Para medir o "realismo" das séries, utilizamos diversas métricas, como o teste Kolmogorov-Smirnov (KS), distância de Jensen-Shannon, distância de Wasserstein, além de gráficos de similaridade T-SNE bidemnsional para verificar visualmente a similaridade distribucional entre dados reais e sintéticos.

### Bases de Dados

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|API do Yahoo Finance| https://finance.yahoo.com | Permite o acesso a dados financeiros por meio de chamadas de API. Esses dados incluem preços de fechamento, preços máximos, mínimos, volume negociado. Além disso, é possível coletar os dados considerando diferentes períodos de amostragem: 2 minutos, 5 minutos, 15 minutos, 1 hora, 1 dia.|


## Metodologia e Workflow
**CASO 1: TRANSFORMERS**

A metodologia para a geração das séries temporais sintéticas utilizando arquitetura Transformers pode ser resumida no seguinte passo a passo:

1. **Coleta de Dados via API do Yahoo Finance:**
   
   Através desse API, coletamos os preços com um período de amostragem de 2 minutos, e armazenamos em um vetor que representa a série temporal: $X\_{1:N}$.
   
   O período de amostragem de 2 minutos foi escolhido pois é o menor que o API disponibiliza. Optamos por realizar uma análise em alta frequência, pois as variações nos preços não são tão abruptas comparadas à de uma frequência menor (e.g. valores diários). Dessa forma, o modelo consegue gerar dados dentro de uma faixa razoável de valores. A figura abaixo ilustra um exemplo.
   
<div align="center">
    <img src="Valores_Vale.png" alt="Preços_Vale" title="Vale" />
    <p><em>Figura 1: Preços das ações da Vale com um período de amostragem de 2 minutos coletados do API do Yahoo Finance.</em></p>
</div>

2. **Extração de Features:**

   Para auxiliar na geração de dados sintéticos realistas, também extraimos diversos features que ajudam a explicar o comportamento dos preços. Esses features também são séries temporais, dados (cada um) por: $F\_{1:N}$. Eles possuem o mesmo número de amostras da série temporal de preços.

Os features que se mostraram úteis na geração dos dados sintéticos foram:

   - Volume de ações negociada;
   - Índices técnicos: Moving Average Convergence Divergence (MACD), Stochastic Oscillator (SO), Commodity Channel Index (CCI), Money Flow Index (MFI);
  
Os índices técnicos são algumas métricas que podem ser calculadas a partir do preço de fechamento, preço máximo e mínimo, além do volume de ações negociadas. Esses índices técnicos buscam capturar as tendências de movimentação dos preços. A figura abaixo ilustra um exemplo de um feature utilizado:

<div align="center">
    <img src="Volume_Vale.png" alt="Volume_Vale" title="Vale" />
    <p><em>Figura 2: Volume de ações da Vale negociadas com um período de amostragem de 2 minutos coletados do API do Yahoo Finance.</em></p>
</div>

3. **Normalização dos Dados:**

   Após a coleta dos dados e extração dos features, armazenamos as séries temporais (do preço e dos features) em um mesmo dataframe: $D=[X\_{1:N}, F\_{1:N} ]$.
   
   Após isso, normalizamos os valores de cada série temporal para facilitar o treinamento, utilizando as suas respectivas médias e desvios padrões. A normalização adotada foi:

$$ x_{n}(i) = \frac{x(i) - \text{média[x]}}{\text{desvio padrão[x]}}$$

- $x_{n}(i)$: representa o valor normalizado de uma série temporal (preço ou algum feature) no instante $i$.
-  $x(i)$: representa o valor antes da normalização (preço ou algum feature) no instante $i$.
- média[x], desvio padrão [x] : representam a média e o desvio padrão associado à série temporal dos elementos de x(i)  
   
4. **Construção da Rede Neural:**
   A rede neural é um modelo baseado na arquitetura Transformer sendo utilizado para predição de séries temporais. Ele processa sequências de dados para predizer o valor futuro com base nas observações passadas. A figura abaixo ilustra o modelo, de maneira simplificada, atráves de blocos:
   <div align="center">
    <img src="Arquitetura_Blocos.png" alt="Arquitetura" title="Arquitetura" />
    <p><em>Figura 3: Estrutura simplificada do modelo baseado na arquitetura Transformer. </em></p>
</div>

- **Input:**
   
   A entrada é um dataframe D contendo a série temporal do preço $X_{1:N}$ e dos features $F\_{1:N}$.
   
- **Sequenciador das Séries Temporais:**
   
   As séries temporais são repartidas em sequências de tamanho fixo (tam_seq) para o processamento nos blocos Transformers. Além disso, associamos a cada sequência um target, que representa o valor que desejamos prever (rótulo). Para o treinamento, a rede recebe um conjunto de sequências e os rótulos correspondentes.
   
- **Layer de Input:**
   
   A entrada da rede é um vetor multidimensional que contém todas as sequências de tamanho tam_seq para todos os features.
   
- **Embedding Layer:**

   A embedding layer é uma camada densa que transforma os dados em um espaço dimensional maior. É útil para que o modelo aprenda relações mais complexas nos dados.

- **Positional Encoding:**

   Adiciona informações sobre a posição de cada elemento da sequência, visto que o Transformer não conhece a ordem temporal dos dados. Isso permite que o modelo saiba a ordem temporal dos dados.

- **Blocos Transformers:**

   Sequências de blocos da arquitetura Transformer, cada bloco possui os seguintes elementos:

   - Layer MultiHead Attention: permite que o modelo se concentre em diferentes partes da sequência para realizar a predição
   - Conexão Residual e Normalização: adiciona a entrada do bloco à saída do layer MultiHead Attention e normaliza os valores. Isso ajuda na estabilização de treinamento.
   - Rede Feed-Forward: duas camadas densas com função de ativação ReLU na primeira.
     
- **Global Average Pooling:**
    
   Reduz a saída dos blocos transformers para um vetor de tamanho fixo através do cálculo da média dos valores.

- **Output Layer**:

    Camada densa que gera o valor predito. No nosso modelo, predizemos apenas um único valor por vez.

Os detalhes específicos da constituição de cada bloco estão descritos neste link: [Detalhes_Arquitetura](docs/Arquitetura.md)

5. **Treinamento:**

Após a construção do modelo, partimos para a etapa de treinamento. Nesta etapa, o nossos dados de entrada $D = [X_{1:N}, F_{1:N}]$ são separados em conjunto de treinamento, validação e teste:

- Conjunto de treinamento: Os 70% primeiros elementos do dataset de entrada
- Conjunto de validação:      20% dos elementos do dataset
- Conjunto de teste:       Os 10% últimos elementos do dataset de entrada

Por exemplo, se o dataset de entrada são séries temporais com 1000 elementos, então os 700 primeiros elementos são utilizados para treinamento, os 200 elementos seguintes para validação, e os últimos 100 para teste. Foi importante garantir que os dados estejam ordenados, pois apresentam dependências temporais.

Conforme explicado no bloco de sequenciamento das séries temporais, os dados são transformados em sequências de tamanho fixo. No nosso caso, observamos que sequências com 24 instantes de tempo consecutivos apresentaram os melhores resultados. Logo, o modelo recebe como entrada sequências com 24 elementos consecutivos e o rótulo associado, que no caso, seria o 25º elemento.

Ou seja, dado os últimos 24 preços (e features), o modelo tentará prever o 25º preço, e a verificação da qualidade da solução será dado pela comparação com o valor do rótulo que é o valor real do 25º preço.

Para o treinamento, foi utilizado os seguintes hiperparâmetros:
- Otimizador: Adaptative Moment Estimator (Adam);
- Função de perda: Mean Absolute Error;
-  Batch size: 128;
-  Número de épocas: 200 (com early stopping);

  A escolha dos melhores parâmetros foi baseado na perda observada para o conjunto de validação.

  6. **Inferência:**

Após o treinamento, utilizamos o modelo para prever os pontos do conjunto de teste e comparamos com os respectivos rótulos associados.

A figura abaixo ilustra o workflow:

 <div align="center">
    <img src="Workflow.png" alt="Workflow" title="Workflow" />
    <p><em>Figura 4: Workflow contemplando o processo de treinamento e inferência. </em></p>
</div>

## Experimentos, Resultados e Discussão dos Resultados




### Artigos de Referência
Os principais artigos que o grupo já identificou como base para estudo e planejamento do projeto são:

- **Pagnocelli. (2022)**: "A Synthetic Data-Plus-Features Driven Approach for Portfolio Optimization" [5].
  
- **Peña et al. (2024)**: "A modified CTGAN-plus-features-based method for optimal asset allocation" [2].

-  **F.Eckerli, J.Osterrieder.** "Generative Adversarial Networks in finance: an overview" [3]. 

### Ferramentas
Existem diversas bibliotecas Python disponíveis para geração de dados sintéticos, cada uma com suas capacidades e recursos distintos. Neste trabalho exploraremos as seguintes bibliotecas CTGAN  e Synthetic Data Vault (SDV).

- **CTGAN** é uma coleção de geradores de dados sintéticos baseados em Deep Learning para dados de tabela única, que são capazes de aprender com dados reais e gerar dados sintéticos com alta fidelidade. 

- **SDV (Synthetic Data Vault)** O pacote é focado na geração e avaliação de dados sintéticos tabulares, multitabelas e séries temporais. Aproveitando uma combinação de modelos de aprendizado de máquina, o SDV fornece recursos e síntese de dados, ao mesmo tempo em que garante que os conjuntos de dados gerados se assemelhem aos dados originais em estrutura e propriedades estatísticas. 

- **Python** com bibliotecas como `PyTorch` e `scikit-learn` para implementar os modelos generativos e realizar a síntese de dados.
   
- **Colab** para colaboração e execução de experimentos em ambientes com suporte a GPU.
  
- **Pandas** e **NumPy** para manipulação de dados tabulares.

### Proposta de Avaliação
Para a avaliação da qualidade dos nossos geradores de dados sintéticos, além dos fatos estilizados, vamos considerar várias outras métricas utilizando amostras reais e sintéticas. As métricas de avaliação que pretendemos utilizar são:

Comparação entre as distribuições sintéticos e históricos usando métricas que capturam os aspectos distribucionais dos dados sintéticos com relação às amostras reais. Neste caso vamos usar o teste Kolmogorov-Smirnov (KS), teste Qui-quadrado (CS) que medem a similaridade para variáveis ​​contínuas e categóricas (colunas) respectivamente. A medidas de divergência distribucional como distância de Jensen-Shannon, Discrepância Média Máxima (MMD) e distância de Wasserstein. Gráficos de similaridade T-SNE bidemnsional para verificar visualmente a similaridade distribucional entre dados reais e sintéticos. 

## Conclusão
Por fim, a principal dificuldade do projeto será gerar os dados financeiros sintéticos realistas. Abordaremos diversas estratégias que vão desde o pré-processamento dos dados, ajustes nos hiperparâmetros das GANs e o emprego de métricas eficientes.
 
## Referências Bibliográficas
[1] Li, Gaorong, Lei Huang, Jin Yang, and Wenyang Zhang.  
"A synthetic regression model for large portfolio allocation."  
*Journal of Business & Economic Statistics* 40, no. 4 (2022): 1665-1677.

[2] Peña, José-Manuel, Fernando Suárez, Omar Larré, Domingo Ramírez, and Arturo Cifuentes. 
"A modified CTGAN-plus-features-based method for optimal asset allocation".
" Quantitative Finance 24, no. 3-4 (2024): 465-479".

[3] https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

[4] F.Eckerli, J.Osterrieder.
" Generative Adversarial Networks in finance: an overview."

[5]- Bernardo K. Pagnoncelli, Arturo Cifuentes, Domingo Ramírez and Hamed Rahimian.
 "A Synthetic Data-Plus-Features Driven Approach for Portfolio Optimization".
 Computational Economics, 2023, Volume 62, Number 1, Page 187.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
