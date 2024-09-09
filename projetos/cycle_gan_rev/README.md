# `Cycle-GAN Revisitado`
# `Revisiting Cycle-GAN`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

|Nome  | RA | Especialização|
|--|--|--|
| Gabriel Freitas  | 289.996  | Eng. ???????|
| Tiago Amorim  | 100.675  | Eng. Civil / Petróleo |

## Descrição Resumida do Projeto

Dentro da visão computacional a **transferência de estilo** busca criar novas imagens combinando o conteúdo de uma imagem com o estilo de outra imagem (ou conjunto de imagens). Esta área de estudo pode ter diferentes aplicações, de transformar cavalos em zebras, fotos em pinturas, esboços em figuras completas ou imagens noturnas em diurnas.

Em 2017 foi apresentada a arquitetura CycleGAN, que conseguiu realizar transferência de estilo em base de dados não pareadas. A partir deste momento foram propostas outras soluções para este problema, alcançando resultados cada vez melhores.

Muitas das arquiteturas mais recentes se baseiam no uso e/ou ajuste fino de modelos de larga escala pré-treinados. O treinamento destas redes requer grande poder computacional e significativo número de amostras.

Este projeto pretende, a partir da arquitetura original da CycleGAN, avaliar o impacto da incorporação de algumas das ideias que foram propostas posteriormente. O objetivo é ter uma arquitetura que possa ser treinada com uma estrutura de hardware mais acessível.

Apresentação da proposta: [slides](https://docs.google.com/presentation/d/1kkJbaO5Ldz5YJYXXRCdzqwaxpyK8gpK8tgvfre5GHNw/edit?usp=sharing).

## Metodologia Proposta
> Para a primeira entrega, a metodologia proposta deve esclarecer:
> * Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.
> * Quais abordagens de modelagem generativa o grupo já enxerga como interessantes de serem estudadas.
> * Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projeto
> * Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).
> * Resultados esperados
> * Proposta de avaliação dos resultados de síntese

## Cronograma
| Semanas | Etapa | Detalhamento |
|--       |-- |--|
| 1-2     | Conhecimento | Busca por referências adicionais.<br> Estudo e adaptação do código original. |
| 3-4     | Experimentos I | Testes com primeiras modificações:<br> - Adaptadores LoRA,<br> - _Skip connections_,<br> - Funções de perda adicionais. |
| 5-6     | Experimentos II | Inclusão de camadas _Transformers_. |
| 7-8     | Novas transformações | A serem definidas. |
| 9-10    | Finalização  | Organização e empacotamento de código.<br> Confecção de relatório.  |

## Referências Bibliográficas

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros. In ICCV 2017.<br>
[[Paper]](https://arxiv.org/abs/1703.10593) [[Github]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

StyTr²: Image Style Transfer with Transformers.<br>
Yingying Deng, Fan Tang, Weiming Dong, Chongyang Ma, Xingjia Pan, Lei Wang, Changsheng Xu. IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2022.<br>
[[Paper]](https://arxiv.org/abs/2105.14576) [[Github]](https://github.com/diyiiyiii/StyTR-2)

One-Step Image Translation with Text-to-Image Models.<br>
Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, Jun-Yan Zhu. In arXiv 2024.<br>
[[Paper]](https://arxiv.org/abs/2403.12036) [[Github]](https://github.com/GaParmar/img2img-turbo)
