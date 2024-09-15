# `Síntese de Dados Financeiros para Otimização de Portfólio`
# `Financial Data Synthesis for Portfolio Optimization`
## Link dos slides
https://docs.google.com/presentation/d/1bETsdaZOJDIBiyRV0t87tP7LU9r7ZH9-kDMQtxbuoEY/edit#slide=id.g2d2fd855179_0_106

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação IA376N - Deep Learning aplicado a Síntese de Sinais, oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

 |Nome  | RA | Curso|
 |--|--|--|
 |José Carlos Ferreira  | 170860  | Eng. Elétrica |
 |Byron Alejandro Acuña Acurio  | 209428  | Eng. Elétrica |

## Descrição do Problema/Motivação
Considere o caso de um investidor que tem acesso a $n$ classes de ativos, cada uma representada por um índice de preço adequado. Definimos o problema de otimização de portfólio como um problema de alocação de ativos no qual o investidor busca maximizar o retorno selecionando a quantidade apropriada para cada classe de ativos, mantendo o risco geral do portfólio abaixo de um nível de tolerância predefinido. Pesquisas recentes em otimização de portfólios financeiros exploraram várias abordagens para aprimorar estratégias de alocação de ativos. A geração de dados sintéticos surgiu como um método promissor para aumentar conjuntos de dados financeiros limitados, com estudos propondo modelos de regressão sintética[1], e redes adversárias generativas condicionais modificadas[2]. 

Portanto, neste trabalho vamos fazer síntese de dados tabulares (ou seja, retornos financeiros). O objetivo de usar dados sintéticos é capturar os cenários de retorno potencial que não estavam presentes nos dados históricos, mas são estatisticamente plausíveis.

## Objetivo
 Propor uma solução baseada em redes neurais adversárias (GAN) para gerar dados financeiros sintéticos, preservando e capturando as características principais dos dados reais para otimização de portfólios e outras aplicações financeiras.
## Metodologia Proposta
### Base de Dados Utilizadas
- **API do Yahoo Finance** permite o acesso a dados financeiros por meio de chamadas de API. Esses dados incluem cotações de ações em tempo real, histórico de preços. Neste trabalhos vamos fazer uso do histórico de preços das seguentes ações Johnson Controls International plc (JCI), Target Corporation (TGT), Comcast Corporation (CMCSA), Campbell Soup Company (CPB), Altria Group, Inc.(MO), APA Corporation (APA), Apple Inc.(AAPL), JPMorgan Chase & Co.(JP), Microsoft Corporation (MSFT).
- **Fama-French Datasets** disponivel em [3]. Esta base de dados contém informações sobre fatores de risco sistemático e é amplamente utilizada em estudos de modelagem de retornos financeiros, como no estudo de regressão sintética de Li et al. [1]. Neste dataset temos os seguentes fatores de risco sistemático Market Risk Premium (Mkt-RF), Small Minus Big (SMB), High Minus Low (HML), Risk-Free Rate (RF).
- **Bloomberg Dataset** conforme utilizado no trabalho de Peña et al. [2]. Esta base de dados inclui dados financeiros detalhados e será útil para o estudo de alocação de ativos e geração de cenários sintéticos de retornos. Neste dataset, temos o histórico de preços das seguintes ações:
    - **us_equities**: Refere-se às ações de empresas sediadas nos Estados Unidos. Esse termo cobre uma ampla gama de ações negociadas nas principais bolsas dos EUA, como a NYSE e a Nasdaq.
    - **us_equities_tech**: Refere-se especificamente às ações de empresas de tecnologia nos Estados Unidos. Inclui companhias como Apple, Microsoft, Google, e outras gigantes de tecnologia. Esse subsetor costuma ter volatilidade e crescimento mais altos em comparação com o mercado geral.
    - **global_equities**: Ações de empresas de todo o mundo, abrangendo vários mercados fora dos Estados Unidos. Este grupo inclui tanto economias desenvolvidas quanto emergentes. É uma categoria mais diversificada geograficamente.
    - **em_equities**: Refere-se a ações de mercados emergentes (EM = Emerging Markets). Esses mercados incluem países como Brasil, Índia, China e outros. Eles tendem a ter maior potencial de crescimento, mas também podem ser mais voláteis e arriscados.
    - **us_hy**: High Yield Bonds dos EUA, também conhecidos como "junk bonds". São títulos corporativos de empresas com classificação de crédito inferior a "investment grade" (grau de investimento), oferecendo maiores retornos devido ao maior risco de inadimplência.
    - **us_ig**: Investment Grade Bonds dos EUA, que são títulos de empresas ou governos com alta classificação de crédito, o que implica em menor risco e, geralmente, menor retorno em comparação com os títulos de "high yield".
    - **em_debt**: Dívida de mercados emergentes, que inclui títulos de dívida emitidos por governos ou empresas de países em desenvolvimento. Esses títulos podem oferecer altos retornos, mas também carregam riscos significativos devido à instabilidade econômica ou política.
    - **cmdty**: Commodities, que incluem ativos como petróleo, ouro, prata, e outros recursos naturais. Investir em commodities pode fornecer proteção contra a inflação e diversificação, mas também pode ser volátil.
    - **long_term_treasuries**: Títulos do Tesouro dos EUA com vencimentos de longo prazo, geralmente 10 anos ou mais. Eles são considerados ativos de baixo risco e são sensíveis às mudanças nas taxas de juros. Quando as taxas de juros sobem, o valor desses títulos tende a cair.
    - **short_term_treasuries**: Títulos do Tesouro dos EUA de curto prazo, geralmente com vencimentos de 1 a 3 anos. São considerados extremamente seguros e menos voláteis que os títulos de longo prazo, sendo usados por investidores que buscam preservar capital.

A escolha dessas bases de dados é justificada pelo seu uso comprovado em estudos anteriores sobre otimização de portfólio e síntese de dados financeiros.

### Abordagens de Modelagem Generativa
Entre as abordagens de modelagem generativa que o grupo pretende explorar estão:
- **Redes Adversárias Generativas (CTGAN)**: A abordagem usando GANs não assume uma forma funcional pré-definida para os dados. A rede aprende diretamente a distribuição dos dados reais (tanto marginais quanto condicionais) e gera amostras sintéticas que imitam os dados reais..
  
- **Modelos de Regressão Sintética**: Como proposto por Li et al. [1], esses modelos oferecem uma abordagem mais interpretável para a geração de dados sintéticos, com base em funções matemáticas e modelos estatísticos para prever o comportamento de variáveis dependentes a partir de um conjunto de variáveis independentes..

### Artigos de Referência
Os principais artigos que o grupo já identificou como base para estudo e planejamento do projeto são:

- **Li et al. (2022)**: "A synthetic regression model for large portfolio allocation" [1].
  
- **Peña et al. (2024)**: "A modified CTGAN-plus-features-based method for optimal asset allocation" [2].

### Ferramentas
Existem diversas bibliotecas Python disponíveis para geração de dados sintéticos, cada uma com suas capacidades e recursos distintos. Neste trabalho exploraremos as seguintes bibliotecas CTGAN  e Synthetic Data Vault (SDV).

- **CTGAN** é uma coleção de geradores de dados sintéticos baseados em Deep Learning para dados de tabela única, que são capazes de aprender com dados reais e gerar dados sintéticos com alta fidelidade. 

- **SDV (Synthetic Data Vault)** O pacote é focado na geração e avaliação de dados sintéticos tabulares, multitabelas e séries temporais. Aproveitando uma combinação de modelos de aprendizado de máquina, o SDV fornece recursos e síntese de dados, ao mesmo tempo em que garante que os conjuntos de dados gerados se assemelhem aos dados originais em estrutura e propriedades estatísticas. 

- **Python** com bibliotecas como `PyTorch` e `scikit-learn` para implementar os modelos generativos e realizar a síntese de dados.
   
- **Colab** para colaboração e execução de experimentos em ambientes com suporte a GPU.
  
- **Pandas** e **NumPy** para manipulação de dados tabulares.

### Resultados Esperados
Os principais resultados esperados são:

- Um conjunto de dados sintéticos gerado para complementação das bases financeiras históricas, capaz de capturar variações de retorno plausíveis que não foram observadas nos dados originais.
  
- Análise de como os dados sintéticos podem melhorar as estratégias de alocação de ativos, levando em consideração diferentes níveis de risco.

### Proposta de Avaliação
Para a avaliação da qualidade dos nossos geradores de dados sintéticos, vamos considerar várias métricas utilizando amostras reais e sintéticas. As métricas de avaliação se encaixam nas seguintes categorias principais:

- **Fidelidade**: Comparação entre as distribuições sintéticos e históricos, usando métricas que capturam os aspectos distribucionais dos dados sintéticos com relação às amostras reais. Neste caso vamos usar o teste Kolmogorov-Smirnov (KS), teste Qui-quadrado (CS) que medem a similaridade para variáveis ​​contínuas e categóricas (colunas) respectivamente. A medidas de divergência distribucional como distância de Jensen-Shannon, Discrepância Média Máxima (MMD) e distância de Wasserstein. Gráficos de similaridade T-SNE bidemnsional para verificar visualmente a similaridade distribucional entre dados reais e sintéticos. 

  
- **Utilidade**: Avaliação do desempenho de diferentes estratégias de alocação com e sem os dados sintéticos, medindo métricas de risco-retorno como o índice de Sharpe e o Value-at-Risk (VaR). Treinar modelos de regressão usando dados sintéticos e testando os modelos com dados reais.

## Cronograma
| Etapa                     | Descrição                                      | Duração Estimada |
|----------------------------|------------------------------------------------|------------------|
| Estudo das Bases de Dados   | Análise e pré-processamento dos dados de Fama-French e Bloomberg | 1 semanas        |
| Estudo de Modelos Gerativos | Investigação de modelos como GANs e Regressão Sintética | 3 semanas        |
| Implementação Inicial       | Implementação dos primeiros modelos generativos | 3 semanas        |
| Avaliação Preliminar        | Análise preliminar da qualidade dos dados sintéticos gerados | 2 semanas        |
| Refinamento do Modelo       | Ajustes no modelo com base nos resultados iniciais | 2 semanas        |
| Avaliação Final             | Avaliação completa do modelo e documentação dos resultados | 2 semanas        |

> 
## Referências Bibliográficas
[1] Li, Gaorong, Lei Huang, Jin Yang, and Wenyang Zhang.  
"A synthetic regression model for large portfolio allocation."  
*Journal of Business & Economic Statistics* 40, no. 4 (2022): 1665-1677.

[2] Peña, José-Manuel, Fernando Suárez, Omar Larré, Domingo Ramírez, and Arturo Cifuentes. 
"A modified CTGAN-plus-features-based method for optimal asset allocation.
" Quantitative Finance 24, no. 3-4 (2024): 465-479.

[3] https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
