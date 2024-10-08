
# `Síntese text-to-speech em português brasileiro com variação de sotaque`
# `Text-to-speech synthesis in Brazilian Portuguese with accent variation`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).
 
|Nome  | RA | Especialização|
|--|--|--|
| João Gabriel Teixeira Lima  | 237473  | Eng. de Computação|
| Rita Braga Soares da Silva  | 251627  | Graduação Estatística|

## Resumo (Abstract)

O projeto visa desenvolver um modelo de síntese de fala capaz de gerar amostras com sotaques variados do português brasileiro. A metodologia se baseia no modelo SYNTACC, adaptado para o contexto brasileiro, com o uso da biblioteca Coqui TTS e dados da base CORAA. O projeto envolve etapas de pré-processamento de áudio, extração de embeddings de sotaque e fine-tuning do modelo YourTTS. O objetivo é aumentar a representatividade de diferentes sotaques no cenário TTS, promovendo maior inclusão linguística e diversidade nas aplicações comerciais e sociais.

## Descrição do Problema/Motivação

Técnicas de deep learning vêm promovendo grandes aprimoramentos no campo da síntese de fala e habilitando possibilidades inovadoras, como conversão de voz e TTS zero-shot, além de alcançar maiores níveis de expressividade e melhor representação de diferentes estilos de fala. No entanto, embora modelos de síntese atuais sejam capazes de capturar grande variabilidade na fala, aplicações mainstream dessa tecnologia têm uma tendência a gerarem amostras de fala que se alinham à variedade padrão da língua. Assim, a síntese de fala com variação de sotaques é algo interessante por uma série de fatores.

Do ponto de vista comercial, é possível pensar que a geração de fala semelhante à variedade local dos usuários pode provocar maior penetração e aceitação de produtos que implementam essa tecnologia. Já em relação a políticas linguíticas, a difusão de modelos dessa natureza seria uma ação no sentido de combater estereótipos forjados a partir da visão preconceituosa de que somente a variedade padrão da língua é "correta", além de contribuir para a manutenção de identidades regionais. 

## Objetivo

O objetivo principal deste projeto é implementar um modelo de síntese de fala capaz de gerar amostras com sotaques de diferentes variedades do português brasileiro.

## Metodologia

A metodologia adotada é inspirada pelo modelo de TTS SYNTACC **[1]**, que é capaz de produzir amostras de fala em inglês com sotaque de falantes estrangeiros.

O SYNTACC é um modelo de TTS end-to-end, ou seja, que realiza todas as etapas de conversão texto-fala em um único loop de treinamento. Para isso, integra vários componentes distintos que são otimizados em conjunto. A fim de inserir a variação de sotaque no treinamento, o SYNTACC produz embeddings de sotaque que são concatenados às saídas do seu encoder de texto.

Este trabalho busca reproduzir os procedimentos reportados pelos autores do SYNTACC. Para isso, iremos dispor de ferramentas disponíveis através da biblioteca Coqui TTS **[2]** para realizar tarefas de pré-processamento de áudio e texto, carregamento de modelos e fine-tuning a partir de checkpoints disponíveis do YourTTS, modelo no qual o SYNTACC se baseia. 

Os dados de treinamento foram extraídos da base CORAA **[3]**, que agrega amostras de áudio transcrito de corpora variados, abrangendo variedades do português de Minas Gerais, São Paulo (capital e interior) e Pernambuco.

### Bases de Dados e Evolução

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|NURC Recife (EFs) | [Link](https://fale.ufal.br/projeto/nurcdigital/) | Corpus da variedade de Pernambuco. Foram consideradas apenas gravações da modalidade "elocução formal", que correspondem a monólogos do falante alvo em fala espontânea. As gravações são bastante antigas e muitas têm áudio de baixa qualidade, portanto foram manualmente selecionadas aquelas com qualidade satisfatória. Cada EF tem duração entre 30 minutos e 1 hora.|
| C-ORAL Brasil | [Link](https://www.c-oral-brasil.org/)| Corpus da variedade de Minas Gerais. A duração dos áudios de cada falante varia entre cerca de 3 a 20 minutos. Muitas amostras apresentam ruído intenso.|

#### Considerações importantes
* Modificação em relação a E1: optou-se por não trabalhar com amostras da variedade de São Paulo pois não há como inferir a tag de identidade de falante a partir do modo como a tabela de metadados do CORAA está anotada.
* Serão explorados meios para realizar redução de ruído nas amostras de áudio.
* Todos os áudios serão reamostrados para 16kHz e normalizados para -27dB (configuração do SYNTACC)
* As transcrições devem passar por um tratamento a fim de converter caracteres especiais e numerais para grafemas (ex.: R$2 -> dois reais). Coqui implementa isso nos argumentos do modelo.
* Autores do SYNTACC reportam bons resultados utilizando 3h de áudio de cada sotaque para realizar fine-tuning do YourTTS, portanto essa é a quantidade de dados que iremos empregar na experimentação inicial.
* Para usar os formatadores de dataset já existentes no Coqui, é necessário organizar os diretórios do dataset da maneira exigida por cada formatador. A estrutura adotada foi a seguinte:

```
root_path/
├── txt/                      # Folder containing transcription files
│   ├── p225/                 # Speaker-specific folder
│   │   ├── p225_001.txt      
│   │   ├── p225_002.txt      
│   │   └── ...               
│   └── ...                   
├── wav48/                    # Folder containing audio files
│   ├── p225/                 
│   │   ├── p225_001.wav      
│   │   ├── p225_002.wav      
│   │   └── ...               
│   └── ...
```
### Workflow

![Workflow](https://lh3.googleusercontent.com/d/1ReUO2nS8wlV9Qu4_g4GLrmAKfuDyc6r2)

## Experimentos, Resultados e Discussão dos Resultados

> Na entrega parcial do projeto (E2), essa seção pode conter resultados parciais, explorações de implementações realizadas e 
> discussões sobre tais experimentos, incluindo decisões de mudança de trajetória ou descrição de novos experimentos, como resultado dessas explorações.

A exploração inicial realizada até este momento teve dois principais intuitos:
* Definir a forma de estruturação do dataset e integrá-la ao formatador de dados do Coqui
* Compreender como os diferentes componentes do YourTTS são implementados dentro do Coqui e quais as configurações devem ser usadas a fim de rodar o modelo desejado.

O material de referência adotado foi [este notebook](/projetos/TTS-Sotaques/notebooks/train_yourtts.py). A implementação básica da estrutura do dataset e cálculo de embeddings de falante pode ser encontrada em [```1-datasetStructure_speakerEmbeddings```](/projetos/TTS-Sotaques/notebooks/1_datasetStructure_speakerEmbeddings.ipynb). 

A biblioteca Coqui emprega muitas das etapas intermediárias de processamento de dados de maneira integrada à definição de configuração de datasets e de modelos. A função de extração de embeddings de falante mostra em sua saída uma chamada ao componente Audio Processor, porém não identificamos exatamente qual parte do código interno do Coqui faz isso. É interessante controlar a chamada ao Audio Processor a fim de tornar possível ajustar suas configurações. 

## Conclusão

Até o momento, o projeto avançou em várias frentes, incluindo a estruturação do dataset, o pré-processamento das amostras de áudio e a implementação de um pipeline simples com o modelo que desempenha o papel de `encoder `. 

No entanto, ainda há desafios a serem enfrentados, particularmente em relação ao tratamento de ruído nos áudios selecionados, o que justifica a necessidade de mais pesquisas sobre técnicas eficazes de remoção de ruído, conforme destacado na primeira etapa do fluxo de trabalho.

Os próximos passos incluem a conclusão da etapa de seleção e tratamento de amostras, o que envolverá experimentações com diferentes técnicas de filtragem de ruído para garantir uma qualidade adequada para o treinamento do modelo. Com essa etapa finalizada, será possível avançar para o carregamento e configuração do modelo no Coqui, seguidos do fine-tuning do YourTTS. Em seguida, o modelo será submetido a um forward pass de teste para identificar possíveis erros, ajustando as configurações conforme necessário.

## Referências Bibliográficas
**[1]** NGUYEN, T.-N.; PHAM, N.-Q.; WAIBEL, A. SYNTACC : Synthesizing Multi-Accent Speech By Weight Factorization. In: ICASSP 2023 - 2023 IEEE INTERNATIONAL CONFERENCE ON ACOUSTICS, SPEECH AND SIGNAL PROCESSING (ICASSP), jun. 2023. ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) [...]. [S. l.: s. n.], jun. 2023. p. 1–5. Disponível em: https://ieeexplore.ieee.org/document/10096431/?arnumber=10096431. Acesso em: 9 set. 2024.

**[2]** https://github.com/coqui-ai/TTS

**[3]** CANDIDO JUNIOR, Arnaldo et al. CORAA ASR: a large corpus of spontaneous and prepared speech manually validated for speech recognition in Brazilian Portuguese. Language Resources and Evaluation, v. 57, n. 3, p. 1139-1171, 2023. Disponível em: https://doi.org/10.1007/s10579-022-09621-4. Acesso em: 07 out. 2024.

