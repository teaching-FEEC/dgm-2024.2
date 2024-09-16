# `<PVFlowNet: Modelo IA Generativo Baseado em Fluxo Normalizador Condicional para Criação de Cenários de Geração Fotovoltaica>`
# `<PVFlowNet: Generative AI Model Based on Conditional Normalizing Flow for Solar Power Generation Scenario Creation>`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

|Nome  | RA | Especialização|
|--|--|--|
| Luis Alexsander Oroya Alvarado  | 217045  | Eng. Elétrica|
| Washington Stalyn Alvarez Orbe  | 224189  | Eng. Elétrica|


## Descrição Resumida do Projeto
O projeto tem como tema a geração de cenários de geração fotovoltaica, com o objetivo de fornecer suporte para aplicações relacionadas ao dimensionamento e à operação de unidades de geração presentes em microrredes. 
Esse trabalho está alinhado com as principais linhas de pesquisa do grupo, especialmente no campo da otimização estocástica, que demanda cenários capazes de modelar a variabilidade das fontes de energia renováveis. Dessa forma, identificou-se uma oportunidade de avanço ao combinar técnicas tradicionais de otimização com modelos de geração de cenários baseados em machine learning.

### Motivação 

- **Variabilidade e Incerteza na Geração Fotovoltaica:** A geração de energia solar apresenta uma variabilidade significativa devido a fatores como clima, horário e sazonalidade. Essa incerteza afeta diretamente a capacidade de prever quanto de energia será gerada em um determinado momento. Como consequência, tanto o planejamento quanto a operação de sistemas de geração fotovoltaica tornam-se desafiadores. A ausência de cenários realistas que capturem essa variabilidade pode levar a falhas no atendimento da demanda energética, colocando em risco a confiabilidade da microrrede.

- **Impacto Econômico da Falta de Cenários Precisos:** A falta de cenários confiáveis também impacta financeiramente. No dimensionamento, sistemas subdimensionados geram uma maior dependência de outras fontes, enquanto o superdimensionamento resulta em investimentos elevados sem necessidade. Operacionalmente, decisões baseadas em previsões imprecisas podem resultar no acionamento de geradores de backup ou na compra de energia a preços elevados em momentos de pico. Isso aumenta os custos operacionais e compromete a viabilidade econômica das microrredes, especialmente em sistemas que integram várias fontes renováveis.

### Objetivo principal:
O objetivo do projeto é desenvolver um Conditional Normalizing Flow-based model, condicionado ao mês e à temperatura, para a geração de cenários de geração fotovoltaica.

> [Link para o vídeo de apresentação](https://drive.google.com/file/d/1Tfcvgrx444mZr-ZxBIK41BJ1kSHkao99/view?usp=sharing)

[Link para a apresentação de slides](https://docs.google.com/presentation/d/1wNXzfYHMkaAU3vaD-WrMQe-kaxg19UTsprnQ_qU0eXA/edit?usp=sharing)

## Metodologia Proposta

### 1. Coleta de Dados
- **Dados de Geração Fotovoltaica**: Inicialmente, foi encontrada uma base de dados da Open Power System Data [[1]](#1), que contém dados simulados de geração fotovoltaica (PV) e eólica (vento). No entanto, essa base de dados é limitada por conter apenas dados simulados. Posteriormente, foi obtida uma base de dados meteorológicos fornecida pela NREL (National Renewable Energy Laboratory) [[2]](#2), referente a uma zona específica, que é mais completa e permite selecionar a localização dos dados. Embora essa base não forneça diretamente os valores de geração fotovoltaica, ela contém informações essenciais para calcular a geração a partir da irradiância solar. Os dados foram coletados em intervalos horários e incluem as seguintes características:

  - **Intervalo Temporal**: Coleta em intervalos horários, capturando variações sazonais e diárias que influenciam a geração de energia.
  - **Atributos Selecionados**:
    - **Temperature**: Temperatura ambiente, que afeta a eficiência dos painéis solares.
    - **GHI (Global Horizontal Irradiance)**: Irradiância solar global incidente em uma superfície horizontal, utilizada para estimar a geração fotovoltaica.
    - **Cloud Type**: Tipo de nuvem, que influencia a quantidade de radiação solar recebida.

- **Variáveis Condicionantes**: Além dos dados de geração, variáveis relevantes serão incluídas como condições para os modelos de Normalizing Flows, como **mês** e, em uma segunda etapa, **temperatura**.

### 2. Pré-processamento dos Dados
- **Limpeza dos Dados**: Tratamento de dados ausentes, remoção de outliers e ajustes necessários para garantir a qualidade dos dados.
- **Normalização**: Os dados de geração serão normalizados para garantir que estejam no intervalo adequado para o treinamento do modelo.
- **Divisão do Dataset**: O dataset será dividido em conjuntos de **treinamento**, **validação** e **teste**, com uma proporção de 70%-15%-15%.

### 3. Definição do Modelo
- **Arquitetura do Modelo**:  Será utilizado um modelo de **Conditional Normalizing Flow (CNF)**, uma técnica que transforma uma distribuição base simples em uma distribuição complexa e altamente multimodal. Diferentemente de abordagens como as redes adversariais generativas (GANs) e autoencoders variacionais (VAEs), os Normalizing Flows permitem calcular diretamente a verossimilhança exata dos dados. Isso é possível porque eles aprendem explicitamente a distribuição de probabilidade dos dados, o que também facilita a geração de novos cenários por meio da inversão do flow [[3]](#3). O CNF será capaz de modelar as correlações interdimensionais presentes nos dados de geração fotovoltaica, condicionando-os às variáveis exógenas. Essa abordagem possibilita capturar de forma precisa as complexidades da geração, mantendo uma estrutura probabilística estável, sem as dificuldades de convergência comuns em outros métodos gerativos [[4]](#4).
- **Variáveis Condicionantes**: Inicialmente, a estação do ano será utilizada como a única variável de condição. Após a análise inicial, **temperatura** também será incluída como uma condição adicional para capturar o impacto climático na geração PV.

### 4. Treinamento
- **Função de Perda**: O treinamento será realizado utilizando uma função de perda baseada na maximização da **log-verossimilhança** dos dados condicionados.
- **Otimizador**: Será utilizado um otimizador como **Adam** com taxa de aprendizado adaptativa para garantir a convergência estável do modelo.
- **Hiperparâmetros**: Os hiperparâmetros, como o número de camadas do flow, a dimensão do espaço latente e o tamanho do batch, serão ajustados utilizando uma técnica de validação cruzada para encontrar a configuração ótima.

### 5. Avaliação
- **Métricas de Avaliação**:
  - **Log-Verossimilhança (Log-Likelihood)**: Será utilizada para avaliar a adequação da distribuição gerada pelo modelo em relação aos dados reais, medindo a capacidade do modelo de ajustar-se às probabilidades dos dados observados.
  - **Erro Médio Absoluto Percentual (MAPE)**: Será utilizado para medir a precisão do modelo em termos percentuais, comparando os valores reais de geração fotovoltaica com os cenários gerados.
  - **Erro Quadrático Médio (RMSE)**: Avaliará a magnitude dos erros ao longo do tempo, penalizando grandes desvios.
  - **Dynamic Time Warping (DTW)**: Comparará a similaridade entre as séries temporais reais e geradas, levando em conta possíveis desalinhamentos temporais.
  - **Correlação Temporal**: Será utilizada para verificar se o modelo captura corretamente as correlações temporais e os padrões de geração fotovoltaica ao longo do tempo [[5]](#5).

- **Análise de Cenários Gerados**: Os cenários gerados serão comparados com os dados reais para verificar se as distribuições de probabilidade aprendidas pelo modelo capturam a variabilidade e as correlações presentes nos dados de geração fotovoltaica.

### 6. Ajustes Adicionais
- **Incorporação da Temperatura**: Após o treinamento inicial, o modelo será ajustado para incluir a temperatura como condição. A análise será repetida para avaliar o impacto dessa variável adicional na geração de cenários.

### 7. Aplicação Final
- **Aplicação Externa**: Neste caso, os cenários gerados pelo modelo serão utilizados como entrada para um problema de dimensionamento e operação de uma microrrede. O objetivo será avaliar o desempenho do modelo ao fornecer dados realistas de geração fotovoltaica para otimizar a capacidade e a operação dos recursos energéticos distribuídos. 

## Cronograma

| Nº da Tarefa | Descrição                                                                 | Data Prevista de Finalização | Semanas Entre Etapas |
|--------------|---------------------------------------------------------------------------|------------------------------|----------------------|
| 1            | Leitura de artigos e familiarização com Normalizing Flows                 | 24/09                        | 2 semanas            |
| 2            | Obtenção da base de dados e definição de métricas de avaliação            | 01/10                        | 1 semana             |
| 3            | Desenvolvimento da estrutura inicial do modelo                            | 08/10                        | 1 semana             |
| 4            | Primeiros resultados com cenários de geração fotovoltaica                 | 22/10                        | 2 semanas            |
| 5            | Validação e ajuste preliminar dos resultados                              | 05/11                        | 2 semanas            |
| 6            | Avaliação final do modelo e ajustes necessários                           | 25/11                        | 3 semanas            |

## Referências Bibliográficas
> Apontar nesta seção as referências bibliográficas adotadas no projeto.

<a id="1">[1]</a> : Open Power System Data. *Renewables.ninja PV and Wind Profiles*. Disponível em: https://data.open-power-system-data.org/ninja_pv_wind_profiles/. Acesso em: setembro de 2024.
<a id="1">[2]</a> : National Renewable Energy Laboratory (NREL). National Solar Radiation Database (NSRDB). Disponível em: https://nsrdb.nrel.gov/. Acesso em: setembro de 2024.
<a id="2">[3]</a> : Dumas J, Wehenkel A, Lanaspeze D, Cornélusse B, Sutera A. A deep generative model for probabilistic energy forecasting in power systems: Normalizing flows. Appl Energy 2022;305:117871.
<a id="3">[4]</a> : Winkler, C., Worrall, D., Hoogeboom, E., & Welling, M. (2023). Learning Likelihoods with Conditional Normalizing Flows. arXiv. https://arxiv.org/abs/1912.00042
<a id="4">[5]</a> : E. Cramer, L. R. Gorjão, A. Mitsos, B. Schäfer, D. Witthaut and M. Dahmen, "Validation Methods for Energy Time Series Scenarios From Deep Generative Models," in IEEE Access, vol. 10, pp. 8194-8207, 2022, doi: 10.1109/ACCESS.2022.3141875