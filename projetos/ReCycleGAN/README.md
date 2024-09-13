# `ReCycleGAN: CycleGAN Revisitado`
# `ReCycleGAN: Revisiting CycleGAN`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

|Nome  | RA | Especialização|
|--|--|--|
| Gabriel Freitas  | 289.996  | Eng. **???????**|
| Tiago Amorim  | 100.675  | Eng. Civil / Petróleo |

## Descrição Resumida do Projeto
<!--
Descrição do tema do projeto, incluindo contexto gerador, motivação.
Descrição do objetivo principal do projeto.
Esclarecer qual será a saída do modelo generativo.
Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 5 minutos).
-->

Dentro da visão computacional a `transferência de estilo` busca criar novas imagens combinando o conteúdo de uma imagem com o estilo de outra imagem (ou conjunto de imagens). Esta área de estudo pode ter diferentes aplicações, de transformar cavalos em zebras, fotos em pinturas, esboços em figuras completas ou imagens noturnas em diurnas.

<div>
<p align="center">
<img src='assets/horse2zebra.gif' align="center" alt="Cavalo para zebra" width=400px style="margin-right:10px;">
<img src='assets/day2night_results_crop.jpg' align="center" alt="Dia para noite" width=400px>
</p>
</div>

<p align="center">
  <strong>Exemplos de transferência de estilo.</strong>
</p>

Em 2017 foi apresentada a arquitetura `CycleGAN`, que conseguiu realizar transferência de estilo em base de dados não pareadas. A partir deste momento foram propostas outras soluções para este problema, alcançando resultados cada vez melhores. Muitas das arquiteturas mais recentes se baseiam no uso e/ou ajuste fino de modelos de larga escala pré-treinados. O treinamento destas redes requer grande poder computacional e significativo número de amostras.

Este projeto pretende, a partir da arquitetura original da CycleGAN, avaliar o impacto da incorporação de algumas das ideias que foram propostas posteriormente. O objetivo final é ter uma arquitetura que possa ser treinada com uma estrutura de hardware mais acessível.

**Objetivos do projeto**:

* Avaliar o uso de `novas estruturas` à arquitetura originalmente proposta para a CycleGAN.
* Incorporar `novas métricas` à avaliação quantitativa e qualitativa das saídas da rede.
* Fazer `comparativo` entre as arquiteturas propostas e redes pré-treinadas propostas na literatura.

**Apresentação da proposta**:
[[slides]](https://docs.google.com/presentation/d/1kkJbaO5Ldz5YJYXXRCdzqwaxpyK8gpK8tgvfre5GHNw/edit?usp=sharing)
[[video]](https://link.fake)

## Metodologia Proposta
<!--
Para a primeira entrega, a metodologia proposta deve esclarecer:
* Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.
* Quais abordagens de modelagem generativa o grupo já enxerga como interessantes de serem estudadas.
* Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projeto
* Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).
* Resultados esperados
* Proposta de avaliação dos resultados de síntese
-->

### Base de Dados

Cada arquitura proposta será treinada no problema de transformar imagens de trânsito de `dia para noite` e de noite para dia. Este tipo de transformação pode ser utilizada para, por exemplo, aumentar uma base de imagens para treinamento sistemas de direção autônoma.

A base de dados a ser utilizada é a [Nexet 2017](https://www.kaggle.com/datasets/solesensei/nexet-original), disponibilizada pela [Nexar](https://data.getnexar.com/blog/nexet-the-largest-and-most-diverse-road-dataset-in-the-world/). Esta base de dados contém 50.000 imagens de câmeras automotivas (_dashboard cameras_), com dados anotados de condição de luz (dia, noite, ocaso) e local (Nova York, São Francisco, Tel Aviv, Resto do mundo). Serão utilizadas as imagens da cidade de Nova York para as condições de dia e noite (4.931 imagens de dia e 4.449 imagens de noite).

### Ferramentas

O código será inicialmente baseado na implementação em Pytorch da [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), com foco na produção de uma rede treinada em base de dados não pareados. A partir deste código inicial serão feitas modificações incrementais, avaliando o seu impacto nos resultados da síntese.

Para acompanhar as diferentes redes treinadas será utilizada a plataforma [Weights & Biases](https://wandb.ai/site).

### Referências

Foram levantados dois artigos que propõem redes de geração de imagens com transferência de estilo com estruturas que podem ser testadas com a CycleGAN. Em [2] é proposta uma rede transformers com uma nova forma de codificação espacial e funções de perda baseadas nos resultados da extração de atributos de uma rede VGG19 pré-treinada. Em [3] é apresentada uma rede que modifica uma rede _stable diffusion_ para receber uma imagem de referência, e faz uso de _skip connections_ para minimizar a perda de informação da imagem de entrada.

### Avaliação

A qualidade das saídas da rede serão avaliadas com _inception score_ (**IS**) e _Fréchet inception distance_ (**FID**). As diferentes redes também serão comparadas entre si por meio de avaliação de preferência por usuários. Ainda é preciso buscar novas ideias para a avaliação da qualidade das saídas da rede proposta.

A expectativa do grupo é de conseguir propor uma variante da `Cycle-GAN` que tenha resultados melhores que o do artigo original, e que ao mesmo tempo seja possível realizar o treinamento desta  rede em hardware mais acessível.


## Cronograma
<!--
Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.
-->

| Semanas | Etapa | Detalhamento |
|--       |-- |--|
| 1-2     | Conhecimento | Busca por referências adicionais.<br> Estudo e adaptação do código original. |
| 3-4     | Experimentos I | Testes com primeiras modificações:<br> - Adaptadores LoRA,<br> - _Skip connections_,<br> - Funções de perda adicionais. |
| 5-6     | Experimentos II | Inclusão de camadas _Transformers_. |
| 7-8     | Avaliação comparativa | Comparar imagens geradas com saídas de redes da literatura. |
| 9-10    | Finalização  | Organização e empacotamento de código.<br> Confecção de relatório.  |

## Referências Bibliográficas
<!--
Apontar nesta seção as referências bibliográficas adotadas no projeto.
-->

[1] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros. In ICCV 2017.<br>
[[Paper]](https://arxiv.org/abs/1703.10593) [[Github]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

[2] StyTr²: Image Style Transfer with Transformers.<br>
Yingying Deng, Fan Tang, Weiming Dong, Chongyang Ma, Xingjia Pan, Lei Wang, Changsheng Xu. IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2022.<br>
[[Paper]](https://arxiv.org/abs/2105.14576) [[Github]](https://github.com/diyiiyiii/StyTR-2)

[3] One-Step Image Translation with Text-to-Image Models.<br>
Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, Jun-Yan Zhu. In arXiv 2024.<br>
[[Paper]](https://arxiv.org/abs/2403.12036) [[Github]](https://github.com/GaParmar/img2img-turbo)
