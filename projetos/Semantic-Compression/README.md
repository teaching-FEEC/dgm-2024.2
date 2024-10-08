# `Compressão Semântica`
# `Semantic Compression`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).


> |Nome  | RA | Especialização|
> |--|--|--|
> | Antonio César de Andrade Júnior  | 245628  | Eng. de Computação |
> | Eduardo Nunes Velloso  | 290885  | Eng. Elétrica |


## Descrição Resumida do Projeto

A compressão de dados, e em particular a compressão de imagens, é um componente fundamental para as novas gerações de comunicação móvel.
Aplicações de natureza crítica, como a telemedicina e os carros autônomos, envolvem decisões que precisam ser tomadas imediatamente com base em uma transmissão de imagens contínua, proveniente de vários sensores simultâneamente.
Para viabilizar essas aplicações, uma compressão a taxas extremamente baixas (menos de 0.1 bit por pixel ou bpp) se faz necessária.
Embora nesse caso já não seja possível preservar o conteúdo estrutural das imagens, essas tarefas estão interessadas somente em certos atributos da imagem, os quais constituem o valor semântico embutido na imagem.

A tarefa da compressão semântica, portanto, é a de projetar um codificador capaz de extrair essa informação semântica e um decodificador capaz de gerar uma imagem reconstruída com o mesmo conteúdo essencial.
O projeto de tais decodificadores pode ser desenvolvido por meio de modelos generativos.

Em [[1]](#1), os autores introduziram uma proposta inicial da utilização de GANs para compressão semântica, tanto utilizando segmentação semântica para sintetizar regiões de interesse (por exemplo, para sintetizar fundos de videochamadas) quanto comprimindo apenas uma sequência de bits (por exemplo, para sintetizar detalhes de partes da imagem).
O trabalho de [[2]](#2) propôs uma arquitetura chamada DSSLIC, também utilizando GANs, que combina a utilização dos mapas semânticos com os resíduos da própria compressão como entradas da rede decodificadora.
Já em [[3]](#3), é utilizado um modelo de difusão latente (LDM), codificando um colormap quantizado da imagem original para auxiliar na reconstrução a partir do vetor semântico extraído.

Este projeto tem como objetivo realizar um estudo comparativo de modelos de compressão semântica baseados nos mencionados acima.

[Link para o vídeo de apresentação](https://drive.google.com/file/d/1sVEZiwOKVfSp3zXrToVWvKQgVfOdiWUl/view?usp=sharing)

[Link para a apresentação de slides](https://drive.google.com/file/d/1XXuT1HYH33gW0SCd8A1U8ulIU7ICBDPW/view?usp=sharing)

## Metodologia Proposta

Diante do contexto apresentado, a proposta deste projeto será implementar diferentes abordagens de redes generativas e compará-las de maneira padronizada com um modelo de referência de compressão tradicional e entre si.

Para isso, será primeiramente analisado um modelo simples baseado em GAN sem rede de segmentação, como em [[1]](#1), um modelo mais sofisticado baseado em GANs condicionais (cGANs) para síntese de regiões de interesse da imagem original, e um modelo baseado em LDM como em [[3]](#3).
A extração de informação semântica se dará por uma modificação da arquitetura ResNet de identificação de objetos em cenas, como foi feito pelo CLIP [[4]](#4) e pela PSPNet [[5]](#5).
Além disso, como modelo de referência, será usado um codec de compressão BPG.

Os conjuntos de dados utilizados para treinamento, tanto do codificador quanto do modelo generativo, foram escolhidos com base na presença de rótulos semânticos identificando os objetos presentes em cada imagem.

Em particular, foram identificados as seguintes bases: 
* ADE20K [[6]](#6), com cerca de 25 mil imagens de cenas com mais de 3 mil categorias de objetos, representados por imagens segmentadas com códigos de cores;
* Cityscapes [[7]](#7), com cerca de 5 mil imagens urbanas, com objetos rotulados entre 30 possíveis classes; e
* Coco-stuff [[8]](#8), com mais de 164 mil imagens de cenas diversas, com 182 classes de objetos identificados.

Finalmente, a avaliação se dará de maneira quantitativa através de dois grupos de métricas.
Isso se deve ao compromisso matemático existente entre a **distorção** observada entre valores de pixels individuais e a **percepção** da qualidade da imagem [[9]](#9).
Para medir a distorção entre as imagens reconstruídas e as imagens originais, serão usadas as métricas usuais de PSNR (peak signal-to-noise ratio) e MS-SSIM (multi-scale structural similarity).
Já no caso da percepção, será comparada a representação semântica obtida nas imagens reconstruídas pelo reuso do codificador com a representação semântica da imagem original, através da relação IoU (intersection-over-union) dos vetores latentes encontrados.
Para as GANs, também serão usadas métricas adversárias (e.g. Feature Matching), e para o modelo de difusão, será utilizada a rede MUSIQ [[10]](#9), que computa uma métrica de qualidade da imagem reconstruída.

Em uma avaliação de ordem qualitativa para teste inicial do conceito também serão consideradas imagens da base de dados da Kodak, comumente utilizada para benchmarks de compressão de imagens.

## Cronograma
>| Tarefa          | Data de Finalização     | Número de Semanas| Progresso  |
>|-----------------|------------|-----------|------------|
>| Revisão bibliográfica        | 17/09/2024 | 2 | █████████░ 90% |
>| Desenvolvimento BPG/VVC | 24/09/2024 | 1 | ░░░░░░░░░░ 0% |
>| Desenvolvimento GAN | 08/10/2024 | 2 | ░░░░░░░░░░ 0%  
>| Entrega 2          | 08/10/2024 | 0 | ░░░░░░░░░░ 0% |
>| Rede de segmentação semântica | 15/10/2024 | 1 | ░░░░░░░░░░ 0% |
>| Desenvolvimento cGAN    | 29/10/2024 | 2 | ░░░░░░░░░░ 0%  |
>| Desenvolvimento Difusão    | 12/11/2024 | 2 | ░░░░░░░░░░ 0%  |
>| Análise dos resultados    | 25/11/2024 | 2 | ░░░░░░░░░░ 0%  |
>| Montar entrega 3   | 25/11/2024 | 0 | ░░░░░░░░░░ 0%  |

## Referências Bibliográficas

<a id="1">[1]</a> E. Agustsson, M. Tschannen, F. Mentzer, R. Timofte, and L. Van Gool, “Generative adversarial networks for extreme learned image compression,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2019, pp. 221–231, https://ieeexplore.ieee.org/document/9010721

<a id="2">[2]</a> M. Akbari, J. Liang, and J. Han, “DSSLIC: Deep semantic segmentation-based layered image compression,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), May 2019, pp. 2042–2046, https://ieeexplore.ieee.org/document/8683541

<a id="3">[3]</a> : T. Bachard, T. Bordin and T. Maugey, "CoCliCo: Extremely Low Bitrate Image Compression Based on CLIP Semantic and Tiny Color Map," 2024 Picture Coding Symposium (PCS), Taichung, Taiwan, 2024, pp. 1-5, doi: 10.1109/PCS60826.2024.10566358., https://ieeexplore.ieee.org/document/10566358

<a id="4">[4]</a> : T. Bachard and T. Maugey, "Can Image Compression Rely on CLIP?," in IEEE Access, vol. 12, pp. 78922-78938, 2024, doi: 10.1109/ACCESS.2024.3408651., https://ieeexplore.ieee.org/abstract/document/10545425

<a id="5">[5]</a> H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Jul. 2017, pp. 2881–2890

<a id="6">[6]</a> B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba, "Semantic understanding of scenes through the ADE20K datase," International Journal of Computer Vision, 127(3), 2019, pp. 302-321

<a id="7">[7]</a> M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, "The cityscapes dataset for semantic urban scene understanding," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 3213–3223, https://ieeexplore.ieee.org/document/7780719

<a id="8">[8]</a> H. Caesar, J. Uijlings, and V. Ferrari, "COCO-Stuff: Thing and stuff classes in context," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, https://ieeexplore.ieee.org/document/8578230

<a id="9">[9]</a> Y. Blau and T. Michaeli, “Rethinking lossy compression: The rate-distortion-perception tradeoff,” in Proc. Int. Conf. Mach. Learn., vol. 97, Jun. 2019, pp. 675–685, 

<a id="10">[10]</a> J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang, “Musiq: Multi-scale image quality transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5148–5157, https://ieeexplore.ieee.org/document/9710973

