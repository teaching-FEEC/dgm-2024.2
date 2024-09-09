# `Síntese de Dados Financeiros para Otimização de Portfólio`
# `Financial Data Synthesis for Portfolio Optimization`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação IA376N - Deep Learning aplicado a Síntese de Sinais, oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

 |Nome  | RA | Curso|
 |--|--|--|
 |José Carlos Ferreira  | 170860  | Eng. Elétrica |
 |Byron Alejandro Acuña Acurio  | 209428  | Eng. Elétrica |

## Descrição do Problema/Motivação
Considere o caso de um investidor que tem acesso a $n$ classes de ativos, cada uma representada por um índice de preço adequado. Definimos o problema de otimização de portfólio como um problema de alocação de ativos no qual o investidor busca maximizar o retorno selecionando a quantidade apropriada para cada classe de ativos, mantendo o risco geral do portfólio abaixo de um nível de tolerância predefinido. Pesquisas recentes em otimização de portfólios financeiros exploraram várias abordagens para aprimorar estratégias de alocação de ativos. A geração de dados sintéticos surgiu como um método promissor para aumentar conjuntos de dados financeiros limitados, com estudos propondo modelos de regressão sintética[1], e redes adversárias generativas condicionais modificadas[2]. 

Portanto, neste trabalho vamos fazer síntese de dados tabulares (ou seja, retornos financeiros). O objetivo de usar dados sintéticos é capturar os cenários de retorno potencial que não estavam presentes nos dados históricos, mas são estatisticamente plausíveis.

## Metodologia Proposta
### Base de Dados Utilizadas
- **Fama-French Datasets** disponivel em [3]. Esta base de dados contém informações sobre fatores de risco sistemático e é amplamente utilizada em estudos de modelagem de retornos financeiros, como no estudo de regressão sintética de Li et al. [1].
- **Bloomberg Dataset** conforme utilizado no trabalho de Peña et al. [2]. Esta base de dados inclui dados financeiros detalhados e será útil para o estudo de alocação de ativos e geração de cenários sintéticos de retornos.
A escolha dessas bases de dados é justificada pelo seu uso comprovado em estudos anteriores sobre otimização de portfólio e síntese de dados financeiros.

### Abordagens de Modelagem Generativa
Entre as abordagens de modelagem generativa que o grupo pretende explorar estão:
- **Redes Adversárias Generativas (GANs)**: Uma abordagem comum para a geração de dados sintéticos, com especial interesse em variantes como as **CTGAN** propostas em [2], onde foi incluida informações contextuais, que se mostram promissoras na síntese de dados tabulares complexos como os retornos financeiros.
  
- **Modelos de Regressão Sintética**: Como proposto por Li et al. [1], esses modelos oferecem uma abordagem mais interpretável para a geração de dados sintéticos, combinando suposições estatísticas com análises baseadas em fatores econômicos.

### Artigos de Referência
Os principais artigos que o grupo já identificou como base para estudo e planejamento do projeto são:

- **Li et al. (2022)**: "A synthetic regression model for large portfolio allocation" [1].
  
- **Peña et al. (2024)**: "A modified CTGAN-plus-features-based method for optimal asset allocation" [2].

### Ferramentas
- **Python** com bibliotecas como `PyTorch` e `scikit-learn` para implementar os modelos generativos e realizar a síntese de dados.
   
- **Colab** para colaboração e execução de experimentos em ambientes com suporte a GPU.
  
- **Pandas** e **NumPy** para manipulação de dados tabulares.

### Resultados Esperados
Os principais resultados esperados são:

- Um conjunto de dados sintéticos gerado para complementação das bases financeiras históricas, capaz de capturar variações de retorno plausíveis que não foram observadas nos dados originais.
  
- Análise de como os dados sintéticos podem melhorar as estratégias de alocação de ativos, levando em consideração diferentes níveis de risco.

### Proposta de Avaliação
A avaliação dos resultados da síntese será baseada em:

- **Métricas Estatísticas**: Comparação entre as distribuições de retornos sintéticos e históricos, usando métricas como distância de Wasserstein e testes de Kolmogorov-Smirnov.
  
- **Desempenho de Alocação de Ativos**: Avaliação do desempenho de diferentes estratégias de alocação com e sem os dados sintéticos, medindo métricas de risco-retorno como o índice de Sharpe e o Value-at-Risk (VaR).

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