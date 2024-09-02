# `Geração Sintética de Dados Aplicado a Reconhecimento de Atividades Humanas (HAR)`
# `Synthetic Data Generation for Human Activity Recognition (HAR)`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*,
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

|Nome  | RA | Especialização|
|--|--|--|
| Bruno Guedes da Silva  | 203657  | Eng. de Computação|
| Amparo Díaz  | 152301  | Aluna especial|



## Descrição Resumida do Projeto

Tema do projeto: Geração de dados de sensores para HAR

Contexto gerador: Projeto do HIAAC

### Motivação: 
- **Falta de Dados:** A escassez de dados relevantes e diversos é um desafio significativo para o treinamento e avaliação de modelos de HAR.
- **Heterogeneidade:** A variabilidade nas classes de atividade, na posição dos sensores e nas características das pessoas cria dificuldades para criar um dataset representativo e generalizável.

### Objetivo principal: 
Modelo que gere dados de sensores de acelerômetro e giroscópio (possivelmente expandir para outras modalidades)

### Saída do modelo generativo:
O modelo generativo produzirá amostras de sensores com 6 canais (acelerômetro e giroscópio) em uma janela de 60 unidades de tempo, ou imagens de espectrogramas representando os dados dos sensores.

### link para vídeo de apresentação da proposta do projeto
    video da proposta

## Metodologia Proposta

### Bases de Dados a Serem Utilizadas

Neste projeto, pretende-se utilizar datasets de ambientes controlados e não controlados para realizar uma comparação entre a performance do modelo generativo em cada cenário.
#### Bases de Dados em ambente controlado
Primeiramente, iremos utilizar o dataset **MotionSense**, escolhido pela sua simplicidade e características:

- **Atividades:** 6 (dws: downstairs, ups: upstairs, sit: sitting, std: standing, wlk: walking, jog: jogging)
- **Participantes:** 24
- **Frequência de Amostragem:** 50Hz
- **Ambiente Controlado:** Sim
- **Citações:** 190 (Scholar)

#### Bases de Dados em ambente não controlado

Após isso, queremos analisar (se possível) o comportamento do modelo também em outros datasets, como o **ExtraSensory**, que é um dataset bem complexo e desbalanceado com muitas atividades:

- **Atividades:** 51
- **Participantes:** 60
- **Frequência de Amostragem:** 40HZ
- **Ambiente Controlado:** Não
- **Citações:** 324 (Scholar)


### Abordagens de Modelagem Generativa Interessantes para HAR

A geração de dados sintéticos para reconhecimento de atividades humanas (HAR) tem sido um campo de grande interesse, especialmente com a utilização de modelos generativos. Tradicionalmente, muitas pesquisas na literatura têm utilizado Redes Adversariais Generativas (GANs) para criar dados sintéticos. No entanto, novas abordagens estão ganhando destaque e podem oferecer vantagens adicionais. Abaixo, apresentamos algumas abordagens de modelagem generativa que poderiam ser relevantes para o contexto de HAR:

#### Redes Adversariais Generativas (GANs)

 Essa abordagem tem sido amplamente utilizada para gerar dados de sensores sintéticos, oferecendo uma boa capacidade de criar amostras realistas.
- **Vantagens:** Capacidade de gerar dados complexos e variados, o que pode ser útil para criar um dataset diversificado de HAR.
- **Desvantagens:** Treinamento pode ser instável e requer um grande número de exemplos para atingir uma boa performance.

#### Modelos de Difusão

- Recentemente, esses modelos têm mostrado resultados promissores em várias tarefas de geração de dados.
- **Vantagens:** Capacidade de gerar amostras de alta qualidade e realismo. Eles podem ser particularmente úteis para criar dados sintéticos de HAR que mantêm as características dos dados reais.
- **Desvantagens:** Modelos geralmente são mais complexos e podem exigir mais recursos computacionais para treinamento.

#### Modelos de Linguagem de Grande Escala (LLMs)

-Têm sido adaptados para tarefas de geração de dados além de processamento de texto. Esses modelos podem ser utilizados para gerar dados sintéticos de sensores ao aprender padrões complexos de grandes conjuntos de dados.

- **Vantagens:** Capacidade de aprender e gerar padrões complexos a partir de grandes volumes de dados, podendo potencialmente capturar nuances nos dados de sensores.
- **Desvantagens:** Modelos geralmente são muito grandes e requerem recursos significativos para treinamento e inferência.

Cada uma dessas abordagens possui características distintas e pode oferecer diferentes vantagens e desafios para a geração de dados sintéticos em HAR.  A decisão final da escolha da abordagem vai ser orientada pela análise dos artigos na área, com preferência pelos trabalhos que utilizam modelos de difusão, visando conhecer melhor o seu funcionamento e potencial para o projeto.

### Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projetos

 [Exploração Inicial de Artigos](https://docs.google.com/spreadsheets/d/11BbbEDeG4cv7K1L7JYR9OeDJNw42FOrAKeA-v5RNAUY/edit?usp=sharing)

Clique no link acima para acessar o documento com a exploração inicial de alguns artigos relevantes para o projeto.

### ### Ferramentas 

Para a geração e avaliação de dados sintéticos em HAR, utilizaremos ferramentas como Google Colab, PyTorch e TensorFlow para treinamento de modelos,Pandas,SciPy e NumPy para análise estatística e manipulação de dados, Keras para desenvolvimento de redes neurais, scikit-learn e pandas para modelagem e preparação de dados, Matplotlib, Seaborn e Plotly para visualização, probavlemente Hugging Face Transformers para processamento de linguagem natural, e ProfileReport junto com table_evaluator  e outtros frameworks para avaliação e comparação detalhada entre dados reais e sintéticos.


### Resultados esperados

### Proposta de avaliação dos resultados de síntese

Para garantir a qualidade e realismo dos dados gerados, serão adotadas várias técnicas de avaliação:

#### ProfileReport
será utilizado para verificar a presença de valores duplicados, incorretos ou inconsistentes no dataset sintético em relação ao dataset real. Essa técnica ajuda a identificar problemas de qualidade nos dados e assegurar que o dataset sintético seja coerente e confiável.


#### Frameworks Especializados para Comparação de Dados Sintéticos e Reais
Frameworks especializados, como o **table_evaluator**, serão empregados para realizar uma comparação qualitativa e quantitativa entre as distribuições dos dados reais e sintéticos. Essas ferramentas permitem uma análise visual detalhada das semelhanças e diferenças, assegurando que o dataset sintético reflita corretamente as características e padrões do dataset original.

#### Métricas Estatísticas

- **STest:** O **STest** será aplicado para avaliar a similaridade entre as distribuições dos dados reais e sintéticos. Esta métrica fornece uma quantificação da correspondência entre os dois conjuntos de dados.
- **LogisticDetection e SVCDetection:** Serão utilizados para quantificar a capacidade de distinguir entre dados reais e sintéticos. Resultados que indicam dificuldade em distinguir os dois conjuntos sugerem alta qualidade no dataset sintético.

#### Avaliação com Machine Learning

- **Descrição:** Um modelo simples de classificação será treinado com dados reais e dados sintéticos, utilizando o mesmo conjunto de teste. A acurácia obtida com ambos os datasets será comparada. Semelhanças nas acurácias indicarão que os dados sintéticos preservam bem as relações e padrões do dataset real.

### Avaliação Adicional

Além das técnicas mencionadas, serão utilizadas outras abordagens para uma análise mais abrangente:

- **Análise Visual por Amostragem Local e Global:** Serão realizadas análises visuais para comparar amostras locais e globais dos dados reais e sintéticos, ajudando a entender como os dados sintéticos se comportam em diferentes contextos.
- **Redução de Dimensionalidade (t-SNE):** A técnica de redução de dimensionalidade **t-SNE** será utilizada para visualizar a estrutura dos dados em espaços de alta dimensão, facilitando a comparação visual entre os dados reais e sintéticos.

#### Análise Quantitativa

- **Semelhança R2R, R2S e S2S:** Métricas de semelhança, como **R2R** (Real-to-Real), **R2S** (Real-to-Sintético) e **S2S** (Sintético-to-Sintético), serão aplicadas para avaliar a correspondência entre os diferentes conjuntos de dados.
- **Usabilidade: R2R, S2R e Mix2R:** Métricas de usabilidade, como **R2R**, **S2R** (Sintético para Real) e **Mix2R** (Mistura de Sintético para Real), serão usadas para avaliar como os dados sintéticos se comportam em tarefas de classificação e outras aplicações práticas.


## Cronograma do Projeto

O cronograma é dividido em quatro fases, cada uma com suas respectivas atividades e períodos estimados.

## Metodologia

### Fase 1: Estudo de Artigos sobre Geração de Sinais Temporais
**Período:** 02/09 a 23/09

**Objetivo:** Revisar e compreender a literatura existente sobre técnicas e métodos para a geração e análise de sinais temporais.

**Atividades:**
- **01/09 a 07/09:** Identificação e coleta de artigos relevantes. Revisão preliminar para entender conceitos principais.
- **08/09 a 20/09:** Leitura detalhada dos artigos selecionados, focando em metodologias e resultados.
- **21/09 a 23/09:** Resumo das abordagens e técnicas descritas. Organização das informações para análise. Análise crítica dos métodos descritos e identificação de lacunas ou áreas de interesse para aplicação no projeto. Escolha dos artigos com técnicas promissoras para reprodução. Preparação de um plano de reprodução dos experimentos.

### Fase 2: Reprodução de Artigos
**Período:** 24/09 a 08/10

**Objetivo:** Reproduzir pelo menos um artigo selecionado.

**Atividades:**
- **24/09 a 05/10:** Implementação de pelo menos um método descrito nos artigos selecionados. Configuração de ambientes e ferramentas necessárias para a reprodução.
- **06/10 a 08/10:** Elaboração da primeira apresentação.

### Fase 3: Adequação para HAR (Reconhecimento de Atividades Humanas)
**Período:** 09/10 a 26/10

**Objetivo:** Adaptar as técnicas e métodos reproduzidos para o contexto específico de HAR, considerando o dataset MotionSense e Extrasensory.

**Atividades:**
- **09/10 a 12/10:** Organização dos datasets e pré-processamento dos dados.
- **13/10 a 26/10:** Ajuste das entradas ao modelo.

### Fase 4: Avaliações e Comparações
**Período:** 27/10 a 25/11

**Objetivo:** Avaliar o desempenho das técnicas adaptadas e comparar com outras abordagens existentes para medir eficácia e eficiência.

**Atividades:**
- **27/10 a 28/10:** Definição de critérios de avaliação e métricas. Realização das avaliações das técnicas adaptadas usando as métricas estabelecidas.
- **29/10 a 11/11:** Ajustes necessários para obter um bom resultado.
- **12/11 a 25/11:** Análise e comparação dos resultados das avaliações. Preparação da apresentação final com conclusões e recomendações.

### Cronograma

| Fase                                          | 02/09 | 09/09 | 16/09 | 23/09 | 30/09 | 07/10 | 14/10 | 21/10 | 28/10 | 04/11 | 11/11 | 18/11 | 25/11 |
|-----------------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| **Fase 1: Estudo de Artigos sobre Geração de Sinais Temporais** |   X   |   X   |   X   |   X   |      |       |       |       |       |       |       |       |       |
| **Fase 2: Reprodução de Artigos**             |       |       |       |       |   X   |   X   |   X   |   X   |       |       |       |       |       |
| **Fase 3: Adequação para HAR (Reconhecimento de Atividades Humanas)** |       |       |       |       |       |       |       |   X   |   X   |   X   |   X   |       |       |
| **Fase 4: Avaliações e Comparações**          |       |       |       |       |       |       |       |       |   X   |   X   |   X   |   X   |   X   |

