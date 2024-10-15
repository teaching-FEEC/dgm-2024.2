# `Síntese text-to-speech em português brasileiro com variação de sotaque`
# `Text-to-Speech Synthesis in Brazilian Portuguese with Accent Variation`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> Incluir nome RA e foco de especialização de cada membro do grupo. Os grupos devem ter no máximo três integrantes.
> |Nome  | RA | Especialização|
> |--|--|--|
> | João Gabriel Teixeira Lima| 237473 | Eng. de Computação|
> | Rita Braga Soares da Silva  | 251627  | Graduação Estatística|


## Descrição Resumida do Projeto
> Este projeto tem como objetivo o desenvolvimento de um modelo de síntese de fala (TTS) em português brasileiro, capaz de gerar amostras de áudio que reflitam a diversidade dialetal do país. O modelo será treinado para produzir fala natural em diferentes variedades do português brasileiro, levando em consideração aspectos variacionais e dialetológicos presentes na língua.

> Além de contribuir para uma maior qualidade nas sínteses de fala, a incorporação dessas variações regionais oferece vantagens mercadológicas, já que produtos que utilizam essa tecnologia poderão alcançar maior aceitação e penetração ao se aproximarem da fala local dos usuários. Sob o ponto de vista de políticas linguísticas, o projeto também representa um avanço na valorização das identidades regionais, combatendo estereótipos e preconceitos associados à ideia de um "padrão" único e correto de fala.

>  [Slides da apresentação](https://docs.google.com/presentation/d/1NmjyT4Ad_2pce2x3cFSUHFNb1d-HhprC/pub?start=false&loop=false&delayms=3000) e [Vídeo da Apresentação](https://drive.google.com/file/d/1pDhBNA9gajQHGKvqa6L8VIibk5YGJuXP/view)

## Metodologia Proposta
> Para a primeira entrega, a metodologia proposta deve esclarecer:
> * Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.
> * Quais abordagens de modelagem generativa o grupo já enxerga como interessantes de serem estudadas.
> * Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projeto
> * Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).
> * Resultados esperados
> * Proposta de avaliação dos resultados de síntese
> A metodologia deste trabalho é baseada no modelo VITS (Variational Inference Text-to-Speech), um sistema de síntese de fala end-to-end. O VITS utiliza um VAE (Variational Autoencoder) para realizar encodings separados do texto e do espectrograma de áudio. 

> Este trabalho propõe modificações no modelo YourTTS, baseadas no artigo SYNTACC: Synthesizing Multi-Accent Speech by Weight Factorization (ICASSP 2023). As modificações incluem: 
> * Introdução de embeddings de sotaque no nível de caractere, somados ao embedding do texto, o que permite capturar as relações letra/som em diferentes sotaques. 
> * Alteração no módulo de encoder de texto do YourTTS para permitir a inclusão do sotaque durante a síntese.
Um ponto positivo desse processo é que o pré-treinamento do YourTTS já inclui gravações em português brasileiro.

> Serão utilizados três conjuntos de dados principais para o treinamento e fine-tuning do modelo:

> * C-ORAL Brasil (variedade mineira).
> * NURC Recife e SP 2010.
> * CORAA (Corpus de Áudios Anotados): Mais de 200 horas de gravações de áudio com transcrição validada.
> Todas as gravações passarão por um rigoroso processo de pré-processamento que inclui: normalização de volume, reamostragem dos dados de áudio, eliminação de ruídos, dado que alguns dos dados foram capturados em ambientes não controlados ou com equipamentos mais antigos.

> Para a implementação e experimentos, as seguintes ferramentas foram utilizadas:

> * Coqui: Utilizada para carregar os modelos e checkpoints.
> * Pytorch: Utilizada para realizar adaptações no código e treinar os modelos com as modificações propostas.

## Cronograma

> 1. **Estudo de Literatura e Conceitos Básicos**  
   - **Duração:** Semanas 1 e 2  
   - Revisão de literatura sobre síntese de fala TTS, abordagens de sotaque e as técnicas relacionadas ao modelo VITS e YourTTS.

> 2. **Configuração e Preparação de Ferramentas**  
   - **Duração:** Semanas 2 e 3  
   - Instalação e configuração de ferramentas necessárias, como Coqui e Pytorch.  
   - Download e análise dos datasets.

> 3. **Pré-processamento dos Dados**  
   - **Duração:** Semanas 3 e 4  
   - Processamento de gravações de áudio, incluindo normalização de volume, reamostragem e eliminação de ruídos.

> 4. **Implementação do Modelo Básico (Entrega 2)**  
   - **Duração:** Semanas 4 e 5  
   - Configuração e implementação inicial do modelo VITS com as modificações do SYNTACC.

> 5. **Fine-tuning**  
   - **Duração:** Semanas 5 e 6  
   - Ajuste fino do modelo utilizando gravações específicas de sotaques, ajustando parâmetros para maximizar a qualidade de saída.

> 6. **Avaliação Inicial e Ajustes**  
   - **Duração:** Semanas 6 e 7  
   - Avaliação da qualidade do modelo com métricas objetivas como Mel Cepstral Distortion (MCD) e Mel Spectral Distortion (MSD) e de métricas perceptuais.
   - Ajustes no modelo com base nos resultados.

> 7. **Documentação e Avaliação Completa**  
   - **Duração:** Semanas 7 e 8  
   - Geração de gravações sintéticas e comparação com gravações reais para calcular as métricas de distorção.  
   - Documentação detalhada dos resultados e metodologia aplicada.

> 8. **Entrega Final**  
   - **Duração:** Semana 9 e 10  
   - Preparação e submissão da versão final do projeto, com todos os ajustes e documentação completa.
 

## Referências Bibliográficas
> KIM, J.; KONG, J.; SON, J. Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech. arXiv:2106.06103 [cs, eess], 10 jun. 2021. Disponível em: http://arxiv.org/abs/2106.06103. Acesso em: 11 set. 2024.

> CASANOVA, E. et al. YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone. arXiv:2112.02418 [cs, eess], 30 abr. 2023. Disponível em: http://arxiv.org/abs/2112.02418. Acesso em: 9 set. 2024.

> NGUYEN, T.-N.; PHAM, N.-Q.; WAIBEL, A. SYNTACC : Synthesizing Multi-Accent Speech By Weight Factorization. In: ICASSP 2023 - 2023 IEEE INTERNATIONAL CONFERENCE ON ACOUSTICS, SPEECH AND SIGNAL PROCESSING (ICASSP), jun. 2023. ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) [...]. [S. l.: s. n.], jun. 2023. p. 1–5. Disponível em: https://ieeexplore.ieee.org/document/10096431/?arnumber=10096431. Acesso em: 9 set. 2024.

> Y. Zhang, Z. Wang, P. Yang, H. Sun, Z. Wang and L. Xie, "AccentSpeech: Learning Accent from Crowd-sourced Data for Target Speaker TTS with Accents," 2022 13th International Symposium on Chinese Spoken Language Processing (ISCSLP), Singapore, Singapore, 2022, pp. 76-80, doi: 10.1109/ISCSLP57327.2022.10037914.

> R. Badlani et al., "Vani: Very-Lightweight Accent-Controllable TTS for Native And Non-Native Speakers With Identity Preservation," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-2, doi: 10.1109/ICASSP49357.2023.10096613. 



