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

Motivação: Falta de dados | Heterogeneidade (Classe de atividade, posição do sensor, características da pessoa) 

Objetivo principal: Modelo que gere dados de sensores de acelerômetro e giroscópio (possivelmente expandir para outras modalidades)

Saída do modelo generativo: Amostras de sensores com 6 canais (acc:xyz, Gyr:xyz), dentro de uma janela de tamanho 60.

> Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 5 minutos).

## Metodologia Proposta
> Para a primeira entrega, a metodologia proposta deve esclarecer:
> * Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.

Neste projeto pretende-se usar datasets de ambientes controlados e não-controlados, para realizar uma comparação entre a performance do modelo generativo em cada cenário. 

Primeiramente iremos utilizar o dataset MotionSense:
- MotionSense: 
    - Atividades:6 (dws: downstairs, ups: upstairs, sit: sitting, std: standing, wlk: walking, jog: jogging)
    - Participantes: 24
    - Frequência de Amostragem: 50Hz
    - Ambiente controlado: sim
    - Citações: 190 (Scholar)

Após isso queremos analisar (se possível) o comportamento do modelo também em outros datasets:
- ExtraSensory:
    - Atividades: 51
    - Participantes: 60
    - Frequência de Amostragem:
    - Ambiente Controlado: Não
    - Citações: 324 (Scholar)

> * Quais abordagens de modelagem generativa o grupo já enxerga como interessantes de serem estudadas.

Abordagens: Muitos trabalhos da literatura utilizam GANs para geração de dados sintéticos, porém mais recentemente outras abordagens utilizando modelos de difusão e LLMs surgiram.

> * Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projetos

**GAN**

Chan, M. H.; Noor,M. H. M. A unified generative model using generative adversarial network for activity recognition. Journal of Ambient Intelligence and Humanized Computing. 2020

**Modelos de difusão**

Unsupervised Statistical Feature-Guided Diffusion Model for Sensor-based Human Activity Recognition

DiffAR: Adaptive Conditional Diffusion Model for Temporal-augmented Human Activity Recognition

> * Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).

Ferramentas: Colab, PyTorch, SciPy

> * Resultados esperados

> * Proposta de avaliação dos resultados de síntese

Avaliação: Vamos comparar o modelo usando diferentes datasets e configurações.

Conforme muitos dos artigos pretendemos utilizar:
- Análise qualitativa: 
    - análise visual por amostragem local e global
    - Redução de dimensionalidade (t-SNE)
- Análise quantitativa: 
    - Semelhança R2R, R2S e S2S
    - Usabilidade: R2R, S2R e Mix2R. 

## Cronograma
> Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto

Fases:
- Fase 1 Estudo artigos geração sinais temporais
- Fase 2 Selecção e reprodução artigos
- Fase 3 Adequação para HAR
- Fase 4 Avaliações e comparações


| |17/09|24/09|01/10|08/10|15/10|22/10|29/10|05/11|12/11|19/11|25/11|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Fase 1|X|X|X|X| | | | | | | |
|Fase 2| |X|X|X| | | | | | | |
|Fase 3| | | |X|X|X|X|X| | | |
|Fase 4| | | | | | | | |X|X|X|
