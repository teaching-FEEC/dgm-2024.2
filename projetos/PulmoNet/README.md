# `<PulmoNet: Rede Neuronal Generativa para Imagens Tomográficas Pulmonares>`
# `<PulmoNet: Generative Neuronal Network for Pulmonary Tomographic Images>`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

 |Nome  | RA | Especialização|
 |--|--|--|
 | Arthur Matheus Do Nascimento | 290906 | Eng. Elétrica |
 | Júlia Castro de Paula | 219193 | Eng. Elétrica |
 | Letícia Levin Diniz | 201438  | Eng. Elétrica |


## Descrição Resumida do Projeto
# Descrição do tema do projeto, incluindo contexto gerador, motivação. 
Esse projeto tem como tema a geração de imagens de tomografia computacional da região torácica, juntamente com a máscara de segmentação das vias aéreas.
A segmentação das vias aéreas é uma técnica muito utilizada para monitoramento de pacientes que apresentam problemas respiratórios/pulmonares. Por meio dela é possível acompanhar as alterações que a enfermidade ou que possíveis tratamentos estão causando na estrutura, que podem levar a problemas respiratórios mais graves.
# Descrição do objetivo principal do projeto.
O objetivo principal do projeto é o de gerar imagens de tomografia computacional torácica com alta fidelidade, no intuito de utilizá-las para alimentar redes de segmentação.
# Esclarecer qual será a saída do modelo generativo.
O modelo generativo proposto terá como saída volumes de tomografia computadoriza da região do tórax, ou seja, uma composição 3D da região dos pulmões, que serão formadas por uma sequência de fatias, ou imagens 2D. Além disso, a rede também ira produzir as máscaras das vias aéres pulmonares de cada uma das fatias geradas.
# A saída do modelo será uma série de fatias/imagens, representando a totalidade de um exame tomográfico.
Imagens tomográficas pulmonares são muito relevantes no contexto diagnóstico de enfermidades pulmonares e para mapeamento para a realização de processos operatórios na região. A seguimentação das vias-aéreas tem se mostrado um facilitador nesses processos por automatizar a identificação de tais estruturas. No entanto, as arquiteturas atuais dependem de um grande volume de dados para serem eficientes. Devido...
# porque redes generativas são úteis nesse problema
O corpo humano é extremamente complexo, é muito difícil conseguir um volume de dados suficiente para obter uma real generalização, então até os que possuem grandes bancos de dados acabam tendo dificuldades em casos mais raros, o que pode ser um problema para a aplicação dos sistemas desenvolvidos. Com redes generativas, é possível criar dados de forma a compensar essa escassez, permitindo que essas redes aprendam muito mais detalhes do que utilizando apenas os dados reais.
# Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 5 minutos).

## Metodologia Proposta
> Para a primeira entrega, a metodologia proposta deve esclarecer:

Este projeto usará como inspiração inicial o trabalho desenvolvido em [[1]](#1), o qual propõe duas arquiteturas baseadas em GANs para a síntese de imagens CT pulmonares a partir de máscaras binárias que segmentam a região pulmonar. Das arquiteturas propostas, inspirar-se-á na arquitetura Pix2Pix, na qual o gerador é composto de um encoder que aumenta a profundidade da imagem enquanto diminui suas dimensões, seguido de um decoder que realiza o processo oposto. Tal arquitetura também utiliza conexões residuais. Na arquitetura Pix2Pix, o discriminador é composto por cinco camadas convolucionais, onde as quatro primeiras são seguidas por uma camada de ativação *LeakyReLu*, enquanto a última é seguida de uma função *sigmoide*. 

Apesar de inspirar-se no artigo [[1]](#1), para o desenvolvimento deste projeto será usada a base de dados ATM'22, a qual possui 500 volumes CTs nos quais as vias aéreas estão completamente anotadas, i.e., delimitadas [[2]](#2). Tal base de dados não foi usada no desenvolvimento do projeto em [[1]](#1), mas foi escolhida no presente projeto devido a sua amplitude, a presença de dados volumétricos e em razão das imagens possuírem a delimitação das vias aéreas obtidas através de especialistas. Os volumes da base ATM'22 foram adquiridos em diferentes clínicas e considerando diferentes contextos clínicos. Construída para a realização de um desafio de segmentação automática de vias aéria utilizando IA, a base de dados está dividida em 300 volumes para treino, 50 para validação e 150 para teste. 

Além do artigo [[1]](#1), também serão considerados os trabalhos realizados em [[3]](#3) e [[4]](#4). No primeiro, desenvolseu-se uma GAN condicional para a geração de imagens CT pulmonares a partir de imagens de ressonância magnética. Já no segundo, utiliza-se um modelo baseado em GAN para a segmentação do pulmão em imagens CT que contém anomalias no tecido pulmonar. Apesar dos objetivos de tais trabalhos não serem os mesmos objetivos propostos para o presente projeto, eles servirão de apoio para proposição de modificações na arquitetura, estratégias de treino e de validação de resultados.   

A ferramenta escolhida para o desenvolvimento da arquitetura dos modelos e de treinamento é o PyTorch, em função de sua relevância na área e familiaridade por parte dos integrantes do grupo.
Ademais, para o desenvolvimento colaborativo dos modelos entre os estudantes, opta-se pela ferramenta de programação Google Collaboratory.
Já para o versionamento dos modelos e para ajustar seus hiperparâmetros, decidiu-se pela ferramenta Wandb AI dentre as opções disponíveis no mercado. A ferramenta do GitHub também auxiliará no versionamento dos algoritmos desenvolvidos.

Como resultado desta implementação, espera-se gerar amostras de imagens de tomografias pulmonares em 2D realistas o suficiente para possibilitar a segmentação das vias aéreas.
Caso este resultado se concretize antes do prazo estipulado pelo cronograma e ainda reste tempo para o aprofundamento do projeto, buscar-se-á a geração de imagens 3D de tomografias pulmonares, isto é, espera-se aumentar o escopo do projeto para gerar volumes com a mesma estratégia da síntese de imagens, com as devidas adequações necessárias a esta nova estrutura.

Por fim, para avaliar a qualidade dos resultados obtidos com o modelo de síntese, propõe-se três tipos de avaliação: análise qualitativa, análise quantitativa e análise frente a um benchmark.
No caso da análise qualitativa, os próprios estudantes irão observar os resultados sintéticos, sejam eles imagens e/ou  volumes, e compararão com os dados reais esperados.
Já a análise quantitativa trata de uma avaliação sobre as imagens a partir dos métodos DICE (xx) e SSIM (xx), conforme feito pelo artigo xxx.
Por último, a análise de benchmark (que pode ser considerada um estratégia quantitativa), tem como objetivo passar os dados reais e sintéticos como entrada de uma rede de segmentação já consolidada e, com isto, compara-se ambas as saídas da rede, coletando as seguintes métricas: DICE, precisão e quantidade de ramificações.

## Cronograma
# Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.


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
