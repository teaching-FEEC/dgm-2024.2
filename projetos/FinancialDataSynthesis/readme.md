# `Síntese de Dados Financeiros para Otimização de Portfólio`
# `Financial Data Synthesis for Portfolio Optimization`
## Link dos slides
Adicionar Aqui
## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação IA376N - Deep Learning aplicado a Síntese de Sinais, oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

 |Nome  | RA | Curso|
 |--|--|--|
 |José Carlos Ferreira  | 170860  | Eng. Elétrica |
 |Byron Alejandro Acuña Acurio  | 209428  | Eng. Elétrica |

## Resumo (Abstract)
> Resumo do objetivo, metodologia **e resultados** obtidos (na entrega E2 é possível relatar resultados parciais). Sugere-se máximo de 100 palavras. 

## Descrição do Problema/Motivação
O desenvolvimento de modelos precisos que utilizam dados financeiros é consideravelmente desafiador devido à complexidade inerente desses dados. Em geral, os dados financeiros são não estacionários e seguem distribuições de probabilidade desconhecidas e difíceis de serem estimadas. Apesar dos avanços nos algoritmos de deep learning, que conseguem capturar melhor essas complexidades, a escassez de dados financeiros disponíveis tem sido um fator limitante na construção de métodos robustos [5].

Há um movimento crescente entre pesquisadores para otimizar modelos de machine learning através da incorporação de dados financeiros sintéticos [4]. A geração de dados sintéticos permite melhorar o desempenho de métodos que, até então, apresentavam resultados insatisfatórios ou eram inviáveis na prática devido à falta de dados, além de possibilitar a simulação de eventos raros ou extremos. 

Diversas metodologias têm sido estudadas. As arquiteturas da família Generative Adversarial Networks (GANs) têm mostrado bons resultados em tarefas de geração de imagens e, mais recentemente, estão sendo aplicadas na geração de dados financeiros sintéticos. A criação de dados financeiros que reproduzam o comportamento de dados reais é essencial para várias aplicações, como o problema de otimização de portfólios.

Considere um investidor com acesso a 𝑛 classes de ativos. O problema de otimização de portfólio consiste em alocar esses ativos de modo a maximizar o retorno, escolhendo a quantidade apropriada para cada classe, enquanto mantém o risco do portfólio dentro de um nível de tolerância predefinido. Pesquisas recentes em otimização de portfólios financeiros exploraram diversas abordagens para melhorar as estratégias de alocação de ativos. A geração de dados sintéticos tem se destacado como uma solução promissora para ampliar conjuntos de dados financeiros limitados, com estudos propondo modelos de regressão sintética [1] e redes adversárias generativas condicionais modificadas [2].

Neste trabalho, focamos na geração de dados sintéticos de ativos listados em bolsas de valores (nacionais e internacionais) utilizando uma abordagem baseada em GANs. A geração de dados sintéticos é particularmente útil para capturar cenários de retorno que estão ausentes nos dados históricos, mas são estatisticamente plausíveis.


## Objetivos
> Descrição do que o projeto se propõe a fazer.
> É possível explicitar um objetivo geral e objetivos específicos do projeto.
>
**Objetivo Geral:**
Gerar dados financeiros sintéticos realistas utilizando redes neurais adversárias (GANs). No caso, computaremos os retornos de índices financeiros nacionais e internacionais (e.g. índice Bovespa ou índice S&P 500). Esses índices representam o desempenho de um conjunto representativo de ativos (em geral, ações). O retorno r(t) para um período t é dado pela equação (1):

$$ r(t) =\frac{x(t) - x(t-1)}{x(t-1)}\quad\quad (1)  $$

Sendo x(t) o valor do índice no período t.

**Objetivos Específicos:**

Neste projeto, temos três objetivos específicos.

**1.** Garantir que os dados financeiros sintéticos (retornos) reproduzam as propriedades estatísticas conhecidas como fatos estilizados.

Os fatos estilizados são propriedades estatísticas comuns, observadas empiricamente, entre diversos ativos financeiros em diferentes mercados e períodos [4]. Os principais fatos estilizados são:

  **a)** Heavy tails: a probabilidade de retornos extremos (ganhos ou perdas extremas) é maior do que visto em distribuições normais. Logo, as extremidades da 
  distribuição de retornos tende a ser mais "grossas" do que as observadas em uma curva normal.

  **b)** Assimetria entre ganhos e perdas: perdas extremas tendem a serem mais prováveis do que ganhos extremos, logo a distribuição de retornos tende a ser 
  assimétrica.

  **c)** Agrupamento de volatilidade: a volatilidade mostra autocorrelação, ou seja, períodos de alta volatilidade tendem a ser seguidos por outros de alta    
  volatilidade, e o mesmo ocorre com períodos de baixa volatilidade.

  **d)** Curva Gaussiana: conforme aumentamos o horizonte do tempo, a distribuição de retornos tende a aproximar da curva Gaussiana. 

  **e)** Ausência de autocorrelação nos retornos: os retornos geralmente apresentam pouca ou nenhuma autocorrelação linear. Por exemplo, o fato do retorno r(t-1) ter sido positivo tem pouca influência no retorno r(t).

Podemos avaliar se a distribuição possui "heavy tails" através da métrica kurtosis (K) que mede a tendência da distribuição apresentar valores extremos. E podemos avaliar se a distribuição possui assimetria entre ganhos e perdas através da métrica skewness (S) que mede a assimetria da distribuição em relação à média.
Por exemplo, verificamos os retornos do Índice Bovespa no período entre 2002 e 2024, e temos o seguinte histograma:

![Histograma](histograma.png "Histograma")

Observamos que o Skewness (S) é negativo, indicando que há uma assimetria entre ganhos e perdas, no caso há uma probabilidade maior de perdas extremas do que ganhos extremos. Enquanto o Kurtosis (K) de 7.28 indica que a probabilidade de retornos extremos é maior do que uma distribuição normal sugere (K>3 já garante isso). Portanto, os fatos estilizados a) e b) são verificadas com este histograma. Os outros fatos também podem ser verificados através de métricas adequadas. Dessa forma, para averiguar se um dado sintético é realista, podemos analisar se ele possui tal propriedades.

**2.** Condicionar a geração de dados sintéticos financeiros à diferentes períodos econômicos.

Conforme mencionado por Peña et al. [3], o comportamento dos ativos pode variar de acordo com o período econômico, também conhecidos como regimes de mercado, em que se encontram. Por exemplo, os ativos em geral se tornam mais voláteis em períodos de crise comparado à tempos de estabilidade. 

Dessa forma, pretendemos separar os períodos econômicos em categorias, como: período de alta, de baixa, de normalidade, e condicionar a geração de dados de acordo com cada categoria, obtendo assim, dados sintéticos mais realistas para cada regime de mercado.

Para tal, podemos utilizar técnicas de clusterização nas séries temporais estudadas, agrupando dados que apresentam comportamentos semelhantes e atribuindo variáveis categóricas para cada cluster. Ao final, geramos os dados utilizando as CTGANs (Conditional Tabular Generative Adversarial Networks) que produzem os dados sintéticos de acordo com o regime de mercado, descrito através da variável categórica. 

## Metodologia Proposta
> Descrever de maneira clara e objetiva, citando referências, a metodologia proposta para se alcançar os objetivos do projeto.
> Descrever bases de dados utilizadas.
> Citar algoritmos de referência.
> Justificar os porquês dos métodos escolhidos.
> Apontar ferramentas relevantes.
> Descrever metodologia de avaliação (como se avalia se os objetivos foram cumpridos ou não?).

**ESCREVER SOBRE O PROCESSO DE CLUSTERIZAÇÃO DOS PERÍODOS ECONÔMICOS (MARKET REGIMS) conforme Peña explica**
#Além do mais, podemos incluir outros features relevantes que influenciam o desempenho do índice, como a taxa de juros (U.S. Treasury Yield). Dessa forma, o dado gerado seria uma tupla contendo informações como: (Data, S&P 500, Taxa de Juros).

O maior desafio do projeto será a geração de dados sintéticos realistas, o que exigirá não apenas bons algoritmos, mas também a escolha cuidadosa dos condicionamentos e dos features mais relevantes.

### Bases de Dados e Evolução
> Elencar bases de dados utilizadas no projeto.
> Para cada base, coloque uma mini-tabela no modelo a seguir e depois detalhamento sobre como ela foi analisada/usada, conforme exemplo a seguir.

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|Título da Base | http://base1.org/ | Breve resumo (duas ou três linhas) sobre a base.|

> Faça uma descrição sobre o que concluiu sobre esta base. Sugere-se que respondam perguntas ou forneçam informações indicadas a seguir:
> * Qual o formato dessa base, tamanho, tipo de anotação?
> * Quais as transformações e tratamentos feitos? Limpeza, reanotação, etc.
> * Inclua um sumário com estatísticas descritivas da(s) base(s) de estudo.
> * Utilize tabelas e/ou gráficos que descrevam os aspectos principais da base que são relevantes para o projeto.

- **API do Yahoo Finance** permite o acesso a dados financeiros por meio de chamadas de API. Esses dados incluem cotações de ações em tempo real e histórico de preços.
- **Fama-French Datasets** disponivel em [3]. Esta base de dados contém informações sobre fatores de risco sistemático e é amplamente utilizada em estudos de modelagem de retornos financeiros, como no estudo de regressão sintética de Li et al. [1]. Neste dataset temos os seguentes fatores de risco sistemático Market Risk Premium (Mkt-RF), Small Minus Big (SMB), High Minus Low (HML), Risk-Free Rate (RF).

- **Talvez nao precise explicar cada ativo**
- **Bloomberg Dataset** conforme utilizado no trabalho de Peña et al. [2]. Esta base de dados inclui dados financeiros detalhados e será útil para o estudo de alocação de ativos e geração de cenários sintéticos de retornos. Neste dataset, temos o retorno histórico dos seguintes ativos:
    - **us_equities**: Refere-se as variações percentuais do índice S&P 500 que é composto pelas 500 maiores empresas listadas na bolsa de Nova York.
    - **us_equities_tech**: Refere-se as variações percentuais do índice Nasdaq 100 que é composto pelas 100 maiores empresas de tecnologia listadas na bolsa de Nova York.
    - **global_equities**: Refere-se as variações percentuais do índice Total Stock Market que representa ações de empresas de todo o mundo, abrangendo vários mercados fora dos Estados Unidos. Este grupo inclui tanto economias desenvolvidas quanto emergentes. É uma categoria mais diversificada geograficamente.
    - **em_equities**: Refere-se as variações percentuais do índice Emerging Markets Stock que representa ações de mercados emergentes. Esses mercados incluem países como Brasil, Índia, China e outros. Eles tendem a ter maior potencial de crescimento, mas também podem ser mais voláteis e arriscados.
    - **us_hy**: Refere-se as variações percentuais do índice High Yield Bonds que representa os títulos corporativos de empresas com classificação de crédito inferior a "investment grade" (grau de investimento), oferecendo maiores retornos devido ao maior risco de inadimplência.
    - **us_ig**: Refere-se as variações percentuais do índice Liquid Investment Grade que representa os títulos de empresas ou governos com alta classificação de crédito, o que implica em menor risco e, geralmente, menor retorno em comparação com os títulos de "high yield".
    - **em_debt**: Refere-se as variações percentuais do índice Emerging Markets Bond refentes às dívidas de mercados emergentes, que inclui títulos de dívida emitidos por governos ou empresas de países em desenvolvimento. Esses títulos podem oferecer altos retornos, mas também carregam riscos significativos devido à instabilidade econômica ou política.
    - **cmdty**: Refere-se as variações percentuais do índice Bloomberg Commodities que incluem ativos como petróleo, ouro, prata, e outros recursos naturais. Investir em commodities pode fornecer proteção contra a inflação e diversificação, mas também pode ser volátil.
    - **long_term_treasuries**: Refere-se as variações percentuais do índice Long-Term Treasury relativos à títulos do Tesouro dos EUA com vencimentos de longo prazo, geralmente 10 anos ou mais. Eles são considerados ativos de baixo risco e são sensíveis às mudanças nas taxas de juros. Quando as taxas de juros sobem, o valor desses títulos tende a cair.
    - **short_term_treasuries**: Refere-se as variações percentuais do índice Short-Term Treasury relativos à títulos do Tesouro dos EUA de curto prazo, geralmente com vencimentos de 1 a 3 anos. São considerados extremamente seguros e menos voláteis que os títulos de longo prazo, sendo usados por investidores que buscam preservar capital.

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

### Workflow
> Use uma ferramenta que permita desenhar o workflow e salvá-lo como uma imagem (Draw.io, por exemplo). Insira a imagem nessa seção.
> Você pode optar por usar um gerenciador de workflow (Sacred, Pachyderm, etc) e nesse caso use o gerenciador para gerar uma figura para você.
> Lembre-se que o objetivo de desenhar o workflow é ajudar a quem quiser reproduzir seus experimentos. 

## Experimentos, Resultados e Discussão dos Resultados

> Na entrega parcial do projeto (E2), essa seção pode conter resultados parciais, explorações de implementações realizadas e 
> discussões sobre tais experimentos, incluindo decisões de mudança de trajetória ou descrição de novos experimentos, como resultado dessas explorações.

> Na entrega final do projeto (E3), essa seção deverá elencar os **principais** resultados obtidos (não necessariamente todos), que melhor representam o cumprimento
> dos objetivos do projeto.

> A discussão dos resultados pode ser realizada em seção separada ou integrada à seção de resultados. Isso é uma questão de estilo.
> Considera-se fundamental que a apresentação de resultados não sirva como um tratado que tem como único objetivo mostrar que "se trabalhou muito".
> O que se espera da seção de resultados é que ela **apresente e discuta** somente os resultados mais **relevantes**, que mostre os **potenciais e/ou limitações** da metodologia, que destaquem aspectos
> de **performance** e que contenha conteúdo que possa ser classificado como **compartilhamento organizado, didático e reprodutível de conhecimento relevante para a comunidade**.

Os principais resultados esperados são:

- Um conjunto de dados sintéticos gerado para complementação das bases financeiras históricas, capaz de capturar variações de retorno plausíveis que não foram observadas nos dados originais.
  
- Análise de como os dados sintéticos podem melhorar as estratégias de alocação de ativos, levando em consideração diferentes níveis de risco.

### Proposta de Avaliação
Para a avaliação da qualidade dos nossos geradores de dados sintéticos, vamos considerar várias métricas utilizando amostras reais e sintéticas. As métricas de avaliação se encaixam nas seguintes categorias principais:

- **Fidelidade**: Comparação entre as distribuições sintéticos e históricos, usando métricas que capturam os aspectos distribucionais dos dados sintéticos com relação às amostras reais. Neste caso vamos usar o teste Kolmogorov-Smirnov (KS), teste Qui-quadrado (CS) que medem a similaridade para variáveis ​​contínuas e categóricas (colunas) respectivamente. A medidas de divergência distribucional como distância de Jensen-Shannon, Discrepância Média Máxima (MMD) e distância de Wasserstein. Gráficos de similaridade T-SNE bidemnsional para verificar visualmente a similaridade distribucional entre dados reais e sintéticos. 

  
- **Utilidade**: Avaliação do desempenho de diferentes estratégias de alocação com e sem os dados sintéticos, medindo métricas de risco-retorno como o índice de Sharpe e o Value-at-Risk (VaR). Treinar modelos de regressão usando dados sintéticos e testando os modelos com dados reais.

## Conclusão

> A seção de Conclusão deve ser uma seção que recupera as principais informações já apresentadas no relatório e que aponta para trabalhos futuros.
> Na entrega parcial do projeto (E2) pode conter informações sobre quais etapas ou como o projeto será conduzido até a sua finalização.
> Na entrega final do projeto (E3) espera-se que a conclusão elenque, dentre outros aspectos, possibilidades de continuidade do projeto.
> 
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
