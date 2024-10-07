# `Obtenção de mapas a partir de imagens de satélite`
# `Obtaining maps from satellite images`

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

- **Base de dados**: Inicialmente, a base de dados a ser utilizada será a mesma utilizada pelos autores da Pix2Pix e consiste em 2196 pares de imagens da cidade de Nova Iorque. As imagens possuem dimensões 256x256 e são separadas em conjuntos de treino, teste e validação com 1099, 550 e547 imagens, respectivamente. O dataset encontra-se disponibilizado gratuitamente na [internet](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/). 

    Outras base de dados utilizadas por outros trabalhos até foram encontradas, mas não são disponibilizadas gratuitamente [[2]](#2)[[5]](#5).

- **Abordagens interessantes**: As duas abordagens escolhidas são baseadas en métodos muito bem estabelecidos na área de I2IT e até hoje são usados como *benchmarks* para novos trabalhos:
    1. Pix2Pix (2016)[[3]](#3): Possivelmente o framework mais amplamente adotado em problemas de I2IT, o modelo consiste em uma GAN condicional onde a condição é a imagem de entrada.

    2. MapGen-GAN (2021)[[2]](#2): MapGen-GAN é uma implementação da CycleGAN otimizada para o problema de extração de mapas de imagens de satélite. A CycleGAN, proposta pelos mesmos criadores do Pix2Pix, também é uma arquitetura muito utilizada em I2IT e tem um processo de treinamento diferente, buscando otimizar uma loss de "consistência de ciclo" que procura garantir que a imagem original seja a mais próxima possível da imagem obtida pelo mapeamento inverso da imagem traduzida.

- **Ferramentas**: As ferramentas utilizadas serão as mais utilizadas na comunidade de Deep Learning atualmente:
    - Criação e treinamento dos modelos: PyTorch
    - Monitoramento de métricas: WandB 

- **Resultados esperados**: Devido à quantidade limitada de dados disponíveis e aos dados utilizados não corresponderem a paisagens brasileiras ou próximas das paisagens da Unicamp, é de se esperar um enviesamento do modelo à cidade de Nova Iorque. Esse enviesamento provavelmente será refletido até mesmo em aplicações do modelo a paisagens rurais ou litorâneas. No entanto, espera-se que o modelo  consiga extrair mapas qualitativamente bons ainda que apresentem algumas distorções, principalmente em áreas mais rurais visto que o *dataset* utilizado será de uma região urbana. 

- **Proposta de avaliação dos resultados de síntese**: As métricas de avaliação encontradas na literatura foram:

    - PSNR (Peak signal-to-noise ratio): Uma medida da qualidade geral de um sinal, é uma das métricas de performance mais utilizadas. Dada uma imagem real $Y$ e uma imagem sintética $\bar{Y}$, o PSNR é dado por 

    $$ PSNR = 20 * \log_{10} \frac{MAX_Y}{MSE} $$

    onde 

    $$ MSE = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [Y(i,j) - \bar{Y}(i,j)]^2 $$

    - RMSE: A raíz do erro quadrático médio pixel-a-pixel. Pode ser calculado rapidamente durante o cálculo do PSRN.

    $$ RMSE = \sqrt{MSE} $$

    - SSIM (Structural Similarity Index): Medida de similaridade entre duas imagens. O SSIM marca cada pixel como o centro de um bloco e então SSIM compara três atributos estatísticos de cada bloco - luminância (média) contraste (variância) e estrutura (covariância). O SSIM então é calculado seguindo a fórmula a seguir: 

    $$ SSIM(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy}+c_2)}{(\mu^2_x+\mu^2_y+c_1)(\sigma^2_x+\sigma^2_y+c_2)} $$

    onde $c_1$ e $c_2$ são constantes para manter a estabilidade.

    - ACC (Acurácia de pixel): Dado um pixel $i$ com valor verdadeiro $(r_i, g_i, b_i)$ e valor previsto $(r_i', g_i', b_i')$, se $\max{(|r_i - r_i'|, |g_i - g_i'|, |b_i - b_i'|)} < \delta$, onde será utilizado o valor $\delta=5$ para obter resultados comparáveis à literatura [[2]](#2).

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
