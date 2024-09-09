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

## Metodologia Proposta
> Para a primeira entrega, a metodologia proposta deve esclarecer:
> * Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.
Fama-French Datasets disponivel em [3] usado em [1]
Bloomberg Dataset disponivel em [2]
> * Quais abordagens de modelagem generativa o grupo já enxerga como interessantes de serem estudadas.
> * Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projeto
> * Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).
> * Resultados esperados
> * Proposta de avaliação dos resultados de síntese

## Cronograma
> Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.
> 
## Referências Bibliográficas
[1] Li, Gaorong, Lei Huang, Jin Yang, and Wenyang Zhang.  
"A synthetic regression model for large portfolio allocation."  
*Journal of Business & Economic Statistics* 40, no. 4 (2022): 1665-1677.

[2] Peña, José-Manuel, Fernando Suárez, Omar Larré, Domingo Ramírez, and Arturo Cifuentes. 
"A modified CTGAN-plus-features-based method for optimal asset allocation.
" Quantitative Finance 24, no. 3-4 (2024): 465-479.
[3] https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html