# `<PulmoNet: Rede Neuronal Generativa para Imagens Tomográficas Pulmonares>`
# `<PulmoNet: Generative Neuronal Network for Pulmonary Tomographic Images>`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

 |Nome  | RA | Especialização|
 |--|--|--|
 | Arthur Matheus Do Nascimento | 290906 | Eng. Elétrica |
 | Júlia Castro de Paula | 219193 | Eng. Eletrica |
 | Letícia Levin Diniz | 201428  | Eng. Elétrica |


## Descrição Resumida do Projeto
> Descrição do tema do projeto, incluindo contexto gerador, motivação. 
> Descrição do objetivo principal do projeto.
> Esclarecer qual será a saída do modelo generativo.
Imagens tomográficas pulmonares são muito relevantes no contexto diagnóstico de enfermidades pulmonares e para mapeamento para a realização de processos operatórios na região. A seguimentação das vias-aéreas tem se mostrado um facilitador nesses processos por automatizar a identificação de tais estruturas. No entanto, as arquiteturas atuais dependem de um grande volume de dados para serem eficientes. Devido...
- porque redes generativas são úteis nesse problema
- objetivo: gerar imagens sintéticas de CT pulmonar comparáveis as reais
- saídas: imagem CT e a segmentação das vias-aereas na imagem
> Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 5 minutos).

## Metodologia Proposta
> Para a primeira entrega, a metodologia proposta deve esclarecer:

Este projeto usará como inspiração inicial o trabalho desenvolvido em [[1]](#1), o qual propõe duas arquiteturas baseadas em GANs para a síntese de imagens CT pulmonares a partir de máscaras binárias que segmentam a região pulmonar. Das arquiteturas propostas, inspirar-se-á na arquitetura Pix2Pix, na qual o gerador é composto de um encoder que aumenta a profundidade da imagem enquanto diminui suas dimensões, seguido de um decoder que realiza o processo oposto. Tal arquitetura também utiliza conexões residuais. Na arquitetura Pix2Pix, o discriminador é composto por cinco camadas convolucionais, onde as quatro primeiras são seguidas por uma camada de ativação *LeakyReLu*, enquanto a última é seguida de uma função *sigmoide*. 

Apesar de inspirar-se no artigo [[1]](#1), para o desenvolvimento deste projeto será usada a base de dados ATM'22, a qual possui 500 volumes CTs nos quais as vias aéreas estão completamente anotadas, i.e., delimitadas [[2]](#2). Tal base de dados não foi usada no desenvolvimento do projeto em [[1]](#1), mas foi escolhida no presente projeto devido a sua amplitude, a presença de dados volumétricos e em razão das imagens possuírem a delimitação das vias aéreas obtidas através de especialistas. Os volumes da base ATM'22 foram adquiridos em diferentes clínicas e considerando diferentes contextos clínicos. Construída para a realização de um desafio de segmentação automática de vias aéria utilizando IA, a base de dados está dividida em 300 volumes para treino, 50 para validação e 150 para teste. 

Além do artigo [[1]](#1), também serão considerados os trabalhos realizados em [[3]](#3) e [[4]](#4). No primeiro, desenvolseu-se uma GAN condicional para a geração de imagens CT pulmonares a partir de imagens de ressonância magnética. Já no segundo, utiliza-se um modelo baseado em GAN para a segmentação do pulmão em imagens CT que contém anomalias no tecido pulmonar. Apesar dos objetivos de tais trabalhos não serem os mesmos objetivos propostos para o presente projeto, eles servirão de apoio para proposição de modificações na arquitetura, estratégias de treino e de validação de resultados.   

>* Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).

- Para desenvolvimento da arquitetura dos modelos e treinamento: Pytorch
- Para colaboração entre os integrantes do grupo: Google Colab
- Para histórico de versões e ajustes de hiperparâmetros: Wandb AI

> * Resultados esperados
- Gerar amostras de imagens onde seja possível realizar a segmentação das vias aéreas
- Se tudo der certo... evoluir de imagens para volumes
> * Proposta de avaliação dos resultados de síntese
- Análise qualitativa: observação das imagens/volumes - GT/sintética
- Análise quantitativa: das imagens: DICE, SSIM; 
- Benchmark: segmentação das imagens reais e sintéticas
    - análise quantitativa da segmentação: DICE, precisão, qntdd ramificações

## Cronograma
> Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.


| Nº da Tarefa | Descrição                                                                 | Data Prevista de Finalização | Semanas Entre Etapas |
|--------------|---------------------------------------------------------------------------|------------------------------|----------------------|
| 1            | Leitura de artigos, familiarização com a base de dados e GANs              | 10/09                        |                      |
| 2            | Primeira versão da GAN (inspirada no artigo de referência)                | 24/09                        | 2 semanas            |
| 3            | Estrutura de avaliação bem delimitada                                     | 07/10                        | 2 semanas            |
| 4            | E2                                                                        | 08/10                        | 1 dia                |
| 5            | Primeiros resultados com imagens segmentadas e valores para validação     | 15/10                        | 1 semana             |
| 6            | Fine-tuning e aperfeiçoamento do modelo                                   | 29/10                        | 2 semanas            |
| 7            | Evoluir para redes 3D ou continuar aperfeiçoando o modelo                 | 05/11                        | 1 semana             |
| 8            | E3                                                                        | 25/11                        | 3 semanas            |



## Referências Bibliográficas
> Apontar nesta seção as referências bibliográficas adotadas no projeto.

<a id="1">[1]</a> : José Mendes et al., Lung CT image synthesis using GANs, Expert Systems with Applications, vol. 215, 2023, pp. 119350., https://www.sciencedirect.com/science/article/pii/S0957417422023685.

<a id="2">[2]</a> : Minghui Zhang et al., Multi-site, Multi-domain Airway Tree Modeling (ATM'22): A Public Benchmark for Pulmonary Airway Segmentation, https://arxiv.org/abs/2303.05745.

<a id="3">[3]</a> :  Jacopo Lenkowicz et al., A deep learning approach to generate synthetic CT in low field MR-guided radiotherapy for lung cases, Radiotherapy and Oncology, vol. 176, 2022, pp. 31-38, https://www.sciencedirect.com/science/article/pii/S0167814022042608.

<a id="4">[4]</a> : Swati P. Pawar and Sanjay N. Talbar, LungSeg-Net: Lung field segmentation using generative adversarial network, Biomedical Signal Processing and Control, vol. 64, 2021, 102296, https://www.sciencedirect.com/science/article/pii/S1746809420304158.

Documento com as referências extras identificadas: https://docs.google.com/document/d/1uatPj6byVIEVrvMuvbII6J6-5usOjf8RLrSxLHJ8u58/edit?usp=sharing
