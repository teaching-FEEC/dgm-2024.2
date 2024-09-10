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

O código será inicialmente baseado na implementação em Pytorch da [Cycle-GAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), com foco na produção de uma rede treinada em base de dados não pareados. A partir deste código inicial serão feitas modificações incrementais, avaliando o seu impacto nos resultados da síntese.

Foram levantados dois artigos que propõem redes de geração de imagens com transferência de estilo com estruturas que podem ser testadas com a Cycle-GAN. Em [2] é proposta uma rede transformers com uma nova forma de codificação espacial e funções de perda baseadas nos resultados da extração de atributos de uma rede VGG19 pré-treinada. Em [3] é apresentada uma rede que modifica uma rede _stable diffusion_ para receber uma imagem de referência, e faz uso de _skip connections_ para minimizar a perda de informação da imagem de entrada.

A qualidade das saídas da rede serão avaliadas com _inception score_ (**IS**) e _Fréchet inception distance_ (**FID**). As diferentes redes também serão comparadas entre si por meio de avaliação de preferência por usuários. Ainda é preciso buscar novas ideias para a avaliação da qualidade das saídas da rede proposta.

Será feito um levantamento de referências adicionais em busca de outros elementos a testar e de formas de avaliação da qualidade das saídas da rede generativa.

No início será abordado o problema _clássico_ de transformar cavalos em zebras, e zebras em cavalos. Será utilizado o data set `horse2zebra`: 939 imagens de cavalo e 1177 de zebras do [ImageNet](http://www.image-net.org), utilizando as keywords `wild horse` e `zebra`.

A expectativa do grupo é de conseguir propor uma variante da Cycl-GAN que tenha resultados melhores que o do artigo original, e que ao mesmo tempo seja possível realizar o treinamento desta  rede em hardware mais acessível.


## Cronograma
| Semanas | Etapa | Detalhamento |
|--       |-- |--|
| 1-2     | Conhecimento | Busca por referências adicionais.<br> Estudo e adaptação do código original. |
| 3-4     | Experimentos I | Testes com primeiras modificações:<br> - Adaptadores LoRA,<br> - _Skip connections_,<br> - Funções de perda adicionais. |
| 5-6     | Experimentos II | Inclusão de camadas _Transformers_. |
| 7-8     | Novas transformações | A serem definidas. |
| 9-10    | Finalização  | Organização e empacotamento de código.<br> Confecção de relatório.  |

## Referências Bibliográficas

[1] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros. In ICCV 2017.<br>
[[Paper]](https://arxiv.org/abs/1703.10593) [[Github]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

[2] StyTr²: Image Style Transfer with Transformers.<br>
Yingying Deng, Fan Tang, Weiming Dong, Chongyang Ma, Xingjia Pan, Lei Wang, Changsheng Xu. IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2022.<br>
[[Paper]](https://arxiv.org/abs/2105.14576) [[Github]](https://github.com/diyiiyiii/StyTR-2)

[3] One-Step Image Translation with Text-to-Image Models.<br>
Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, Jun-Yan Zhu. In arXiv 2024.<br>
[[Paper]](https://arxiv.org/abs/2403.12036) [[Github]](https://github.com/GaParmar/img2img-turbo)
