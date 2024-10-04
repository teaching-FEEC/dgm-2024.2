# `Obtenção de mapas a partir de imagens de satélite`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> |Nome  | RA | Especialização|
> |--|--|--|
> | Vinicius Ventura Andreossi  | 195125  | Eng. de Computação|
> | Cosme Rodolfo Roque dos Santos  | 042687  | Doutorado Eng. Elétrica|


## Descrição Resumida do Projeto
Uma das subáreas de IA generativa que obteve alguns dos mais impressionantes resultados dos últimos anos tem sido a área de *image-to-image translation* (I2IT) [[1]](#1). Dentro dessa subárea, um problema frequentemente abordado é a obtenção de mapas a partir de imagens de satélite, e vice-versa, devido às suas inúmeras aplicações, por exemplo, ajudando governos a tomarem medidas rapidamente em casos de desastres naturais [[2]](#2).

A motivação pelo estudo desse problema é estudar aplicações menos convencionais de modelos generativos, visto que a síntese de dados já é estudada com frequência. Além disso, mapas são um tipo de dado muito rico em informações diversas, sendo relevantes para problemas distintos, desde criação de rotas até segmentação semântica de vegetações. 

O objetivo principal do projeto será criar um modelo generativo que recebe em sua entrada uma imagem de satélite qualquer e produz como saída uma imagem de mesma dimensão traduzida para um mapa. O mapa obtido deve preservar aspectos julgados como relevantes para esse tipo de dado, como consistência de ruas, preservação de rotas e identificação de propriedades do terreno como presença corpos d'àgua, parques, etc.

Como objetivo principal do projeto, tentaremos extrair o mapa da Unicamp de sua foto de satélite e testar a generalização do modelo tentando extrair o mesmo mapa a partir de uma imagem de drone.

## Metodologia Proposta

- **Base de dados**: Inicialmente, a base de dados a ser utilizada será a mesma utilizada pelos autores da Pix2Pix e consiste em 2196 pares de imagens da cidade de Nova Iorque. As imagens possuem dimensões 256x256 e são separadas em conjuntos de treino, teste e validação com 1099, 550 e 547 imagens, respectivamente. O dataset encontra-se disponibilizado gratuitamente na [internet](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/). 

    Outras base de dados utilizadas por outros trabalhos até foram encontradas, mas não são disponibilizadas gratuitamente [[2]](#2)[[5]](#5).

- **Abordagens interessantes**: As duas abordagens escolhidas são baseadas en métodos muito bem estabelecidos na área de I2IT e até hoje são usados como *benchmarks* para novos trabalhos:
    1. Pix2Pix (2016)[[3]](#3): Possivelmente o framework mais amplamente adotado em problemas de I2IT, o modelo consiste em uma GAN condicional onde a condição é a imagem de entrada.

    2. MapGen-GAN (2021)[[2]](#2): MapGen-GAN é uma implementação da CycleGAN otimizada para o problema de extração de mapas de imagens de satélite. A CycleGAN, proposta pelos mesmos criadores do Pix2Pix, também é uma arquitetura muito utilizada em I2IT e tem um processo de treinamento diferente, buscando otimizar uma loss de "consistência de ciclo" que procura garantir que a imagem original seja a mais próxima possível da imagem obtida pelo mapeamento inverso da imagem traduzida.

- **Ferramentas**: As ferramentas utilizadas serão as mais utilizadas na comunidade de Deep Learning atualmente:
    - Criação e treinamento dos modelos: PyTorch
    - Monitoramento de métricas: WandB 

- **Resultados esperados**: Devido à quantidade limitada de dados disponíveis e aos dados utilizados não corresponderem a paisagens brasileiras ou próximas das paisagens da Unicamp, é de se esperar um enviesamento do modelo à cidade de Nova Iorque. Esse enviesamento provavelmente será refletido até mesmo em aplicações do modelo a paisagens rurais ou litorâneas. No entanto, espera-se que o modelo  consiga extrair mapas qualitativamente bons ainda que apresentem algumas distorções, principalmente em áreas mais rurais visto que o *dataset* utilizado será de uma região urbana. 

- **Proposta de avaliação dos resultados de síntese**: As métricas de avaliação encontradas na literatura foram:


**1. Erro Quadrático Médio (MSE)**

  **Teoria**:
  O Erro Quadrático Médio é uma métrica comum usada para quantificar a diferença entre a imagem prevista e a imagem de verdade. É definido como:

 ![image](https://github.com/user-attachments/assets/a9ff5edd-390a-424a-884d-766c5534f618)

  onde N é o número total de pixels, Itrue é o valor do pixel i da imagem real e Ipred é o valor do pixel i da imagem gerada.

  **Referências**:

  •	Zhang, Z. et al. (2012). "A survey of image denoising methods." Proceedings of the IEEE.
  •	Shapiro, J. (2001). "Embedded Image Coding using the Coder/Decoder." IEEE Transactions on Image Processing.

  **Aplicação na Tradução de Imagem**:

  Em tarefas de tradução de imagem, o MSE é usado para avaliar a precisão em nível de pixel das imagens geradas. Um MSE mais baixo indica uma correspondência mais próxima com a imagem de verdade, sugerindo que as imagens geradas preservam bem os detalhes e o conteúdo das imagens originais.



**2. Relação Sinal-Ruído de Pico (PSNR)**
   
  **Teoria**:
  O PSNR é derivado do MSE e fornece uma medida do erro máximo. É expresso em decibéis (dB) e definido como:

 ![image](https://github.com/user-attachments/assets/310943d6-47e4-4350-b15b-ffa2d79571cd)

  onde p é o número de bits por pixel.

  **Referências**:

  •	Gonzalez, R. C., & Woods, R. E. (2002). Digital Image Processing. Prentice Hall.
  •	Wang, Z., & Bovik, A. C. (2009). "Mean Squared Error: Love it or Leave it?" IEEE Signal Processing Magazine.

  **Aplicação na Tradução de Imagem**:

  O PSNR é usado em tradução de imagem para fornecer uma maneira padronizada de comparar a qualidade das imagens geradas em relação à verdade. Valores de PSNR mais altos indicam melhor qualidade da imagem, significando menos distorção nas imagens geradas.



**3. Índice de Similaridade Estrutural (SSIM)**

  **Teoria**:
  O SSIM é uma métrica perceptual que mede a similaridade estrutural entre duas imagens. Baseia-se na ideia de que o sistema visual humano é altamente sensível às informações estruturais nas imagens. O SSIM é calculado usando três componentes: luminância, contraste e estrutura:

![image](https://github.com/user-attachments/assets/65b1b8f5-0d5b-4a0f-823c-d399c0b44076)

  onde μ e σ são as médias e desvios padrão das imagens, e C1 e C2 são constantes para estabilizar a divisão.

  **Referências**:

  •	Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). "Image quality assessment: From error visibility to structural similarity." IEEE Transactions on Image Processing.
  •	Wang, Z., & Bovik, A. C. (2006). "Mean Squared Error: Love it or Leave it?" IEEE Signal Processing Magazine.

  **Aplicação na Tradução de Imagem**:

  O SSIM é particularmente útil em tarefas de tradução de imagem, pois fornece uma abordagem mais centrada no ser humano para avaliar a qualidade da imagem. Ao contrário do MSE e do PSNR, que podem ser sensíveis a pequenos erros pixel por pixel, o SSIM captura diferenças perceptuais na  
 estrutura e no padrão, tornando-se uma métrica valiosa para avaliar a qualidade das imagens geradas que devem ser visualmente similares às suas correspondentes de verdade.


   

## Cronograma
TODO: MAKE A COLORED TABLE 
| Data    | Planejamento |
| --------| ------- |
| 17/09   | Definição do tema e definição da bibliografia. |
| 24/09   | Implementação inicial dos modelos + treinamento com conjunto arbitrário de hiperparâmetros. Visualização, análise e pré-processamento dos dados |
| 08/10   | Implementação preliminar concluída com modelos otimizados e produzindo resultados qualitativamente satisfatórios.|
| 22/10   | Implementação das métricas e avaliação quantitativa dos resultados obtidos. |
| 12/11   | Segunda rodada de otimização dos modelos visando melhorar as métricas quantitativas. Possivelmente os modelos podem já estar concluídos nessa data.  |
| 26/11   | Duas semanas de "folga" para resolver eventuais problemas que possam aparecer. README.md escrito explicando em detalhes o funcionamento do projeto e documentando os resultados obtidos. Aplicativo Streamlit / Gradio mostrando o projeto em funcionamento (*). |

(*): Esses recursos só serão implementados caso o projeto esteja funcionando adequadamente e os resultados quantitativos já tenham sido coletados. 

## Referências Bibliográficas
<a id="1">[1]</a>
Hoyez, H.; Schockaert, C.; Rambach, J.; Mirbach, B.; Stricker, D. "Unsupervised Image-to-Image Translation: a Review". *Sensors*. **2022**, 22, 8540

<a id="2">[2]</a>
Jieqiong Song, Jun Li, Hao Chen, and Jiangjiang Wu. "MapGen-GAN: A Fast Translator for Remote Sensing Image to Map Via Unsupervised
Adversarial Learning", in
*IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*. **2021**, 14, 2341.

<a id="3">[3]</a>
P. Isola, J. -Y. Zhu, T. Zhou and A. A. Efros, "Image-to-Image Translation with Conditional Adversarial Networks," 2017 *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Honolulu, HI, USA, 2017, pp. 5967-5976, doi: 10.1109/CVPR.2017.632.

<a id="4">[4]</a>
J. -Y. Zhu, T. Park, P. Isola and A. A. Efros, "Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks," 2017 *IEEE International Conference on Computer Vision (ICCV)*, Venice, Italy, 2017, pp. 2242-2251, doi: 10.1109/ICCV.2017.244.

<a id="5">[5]</a>
Song, J.; Li, J.; Chen, H.; Wu J. RSMT: A Remote Sensing Image-to-Map Translation Model via Adversarial Deep Transfer Learning. *Remote Sensing*. **2022**, 14, 919.
