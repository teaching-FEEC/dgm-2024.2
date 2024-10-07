# `<Redes Adversárias Generativas Sociais: Geração de Amostras Futuras para Predição da Trajetória Humana em Espaços Populados>`



## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> |Nome  | RA | Especialização|
> |--|--|--|
> | Hiuri Santana de Noronha  | 229961  | Eng. Eletricista|
> | Thiago Belina Silva Ramos  | 203975  | Eng. Eletricista|

## Resumo

## Descrição do Problema/Motivação

A predição de trajetórias humanas em ambientes densamente povoados é uma tarefa essencial para que no futuro, sistemas autônomos como carros e robôs possam interagir de forma segura e eficiente com pessoas. É necessário que estes consigam avaliar o ambiente e prever com um grau confiável de precisão as trajetórias futuras dos pedestres a fim de evitar colisões. Entretanto, modelar este comportamento é uma tarefa complexa e que impõe uma série de desafios pois é necessário entender o processo implícito de interações sociais existentes entre seres humanos.

Nesse contexto, as Redes Adversariais Generativas (GANs) surgem como uma possível solução.  As GANs, por serem modelos generativos, são capazes de capturar a incerteza inerente ao movimento humano, gerando múltiplos cenários futuros multimodais, dentre os quais os socialmente aceitáveis serão escolhidos.  A fim de tornar esse processo ainda mais eficiente, foram introduzidos mecanismos sociais neste tipo de rede a fim de permitir que o gerador compreenda melhor tais dinâmicas, criando-se assim as GANs Sociais (SGANs). A principal motivação para a realização deste trabalho é a compreensão desta nova e latente área para aplicações futuras em robótica social (robôs humanoides), cidades inteligentes e sistemas inteligentes de transporte.

 A figura 01 abaixo mostra de forma simplificada e ilustrativa como se dá o processo de predição da trajetória humana em espaços populados, cujas linhas sólidas (azul, vermelha e verde) são a representação do caminho real percorrido pelo pedestre e as linhas tracejadas são a representação de amostras sintéticas multimodais (espaço-tempo) geradas a partir do modelo profundo livres de colisão.

![objetivo](/projetos/HTF/images/FIG01.png)
>Figura 01: Exemplo de predição de trajetória humana em ambientes povoados

Fonte: *Safety-Compliant Generative Adversarial Networks for Human Trajectory Forecasting(Parth Kothari and Alexandre Alahi, 2023)*


> A seguir tem-se os links para o vídeo da apresentação do entregável I e para a apresentação em slides.
 
> [Vídeo da Apresentação](https://drive.google.com/file/d/1NyRet8UhioGTLzvMHryGt_-gbP54u1VB/view) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Link da Apresentação](https://docs.google.com/presentation/d/1vHEd9DeePXOXCwRnCimp_5Cx16KPEKW2/edit?usp=sharing&ouid=101073047652792710630&rtpof=true&sd=true)

## Objetivo

O projeto de pesquisa proposto irá estudar e propor modelos profundos de redes Generativas Adversariais (GANs) para geração de amostras futuras de possiveis trajetórias humanas em espaços populados. Ao combinar a arquitetura tradicional da GAN com auto encoders, células recorrentes Long-Short-Term Memory (LSTM) e módulos de pooling,  espera-se modelar interações multimodais de espaço e tempo de trajetórias de pedestres conforme os acordos sociais implícitos existentes em uma distribuição real desconhecida. O processo adversarial é capaz de gerar o que se denomina de amostragem colaborativa (CS) cujo discriminador (D) possui a tarefa primordial de disciplinar o gerador (G) e garantir que a distribuição de G seja próxima à distribuição real desconhecida. Tal processo é capaz de gerar amostras confiáveis de trajetória humana em espaços populados e cujos modelos profundos devem ser avaliados apropriadamente de acordo com métricas bem estabelecidas na literatura. Apesar da proposta inicial contemplar a implementação de novas estruturas descritas em [1], dentre as quais destacam-se o módulo de interação espacial dos dados de entrada para criar um embedding e o descriminador baseado em arquitetura transformer, este será um objetivo secundário que será trabalhado apenas se os objetivos primários forem alcançado.

## Metodologia

O desenvolvimento e treinamento do modelo terá com ponto de partida e benchmark o repositório disponibilizado em [2]. Portanto, o primeiro passo será sua reprodução, a fim de confirmar os resultados disponibilizados e compreender as propostas realizadas. O repositório permite a avaliação de redes previamente treinadas disponibilizadas, bem como novos treinamentos tanto para dos datasets ultilizados , quantos para outros que respeitem o formato proposto, que são dados tabulares obtidos a partir de videos que contem o número do frame, o número de identificação do pedestre conforme ordem de aparição no vídeo e suas coordenadas x e y. Tais datasets proporcionam ao modelo profundo um conjunto de dados estruturados na forma de séries temporais contendo o posicionamento de pedestres ao longo de uma via a um determinado tempo. Com isso, os modelos profundos podem ser treinados para modelar o conjunto de regras sociais implícitas existentes em trajetórias humanas.

As avaliações qualitativas serão realizadas por observações gráficas que comparam os movimentos reais observados, aos preditos comforme o exemplo da figura xx e as avaliações quantitaivas utilizarão as métricas do benchmark que são o Erro de Deslocamento Médio (ADE - Average Displacement Error), que mede a distância média entre todas as posições previstas e as trajetórias reais ao longo do tempo, fornecendo uma visão geral de quão próximas as trajetórias previstas estão das trajetórias reais dos pedestres. No entanto, o ADE não capta diretamente as interações entre pedestres, algo que modelos como o Social GAN busca melhorar por meio de técnicas como "Social Pooling". Tem-se também o Erro de Deslocamento Final (FDE - Final Displacement Error), que de forma semelhante ao ADE, mede a distância entre a posição final das trajetórias previstas e a posição final real dos pedestres. Essa métrica é particularmente utilizada para avaliar a precisão do modelo ao prever a posição ao final do horizonte de previsão. Assim como o ADE, o FDE, isoladamente, não avalia as interações sociais entre pedestres, sendo complementadas pela Taxa de Colisão, que avalia a porcentagem de trajetórias previstas que resultam em colisões entre pedestres. Esta métrica é fundamental para verificar se o modelo gera trajetórias socialmente aceitáveis. Modelos que geram trajetórias com altas taxas de colisão indicam que o comportamento aprendido não está alinhado com interações sociais naturais, notáveis pela ausência de colisões em geral. No caso de modelos como o Social GAN, a consideração das interações espaciais e temporais é essencial para minimizar essas colisões. Tais métricas podem ser avalidadas considerando o número total de predições geradas, ou ainda pela aplicação do método Top-K, em que são selecionadas apenas trajetórias potenciais para o futuro, e a saída do modelo consiste nas K trajetórias mais prováveis, com base nos padrões e interações aprendidos.


## Datasets:

Para o projeto, serão consideradas duas bases de dados principais conforme tabelas abaixo, constituidas por vídeos de trajetórias humanas no mundo real, com cenários ricos em interações.  A primeira é a BIWI Walking Pedestrians e a segunda a UCY Crowd. Ambos foram convertidos para coordenadas do mundo real em metros, interpolados para obter valores a cada 0,4 segundos.

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|BIWI Walking Pedestrians Dataset | https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz | Vista superior de pedestres caminhando em cenários povoados.|

Esta base de dados é composta por duas cenas, a ETH e a Hotel, cujas imagens podem ser observadas nas figuras 02 e 03 respectiviamente.

<div style="text-align: center;">
    <img src="/projetos/HTF/images/biwi_eth.png" alt="Figura 02: Imagem do dataset ETH" width="600"/>
    <p><em>Figura 02: Imagem do dataset Biwi ETH</em></p>
</div>

<div style="text-align: center;">
    <img src="/projetos/HTF/images/biwi_hotel.png" alt="Figura 03: Imagem do dataset HOTEL" width="600"/>
    <p><em>Figura 03: Imagem do dataset Biwi Hotel</em></p>
</div>

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|UCY Crowd Data | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data | Conjunto de dados contendo pessoas em movimento em meio a multidões.|

Esta base de dados é composta por seis cenas que são Zara01, Zara02, Zara03, Students001, Students003 e Univ. Exemplos de imagens do Zara01 e Students003  podem ser observadas nas figuras 04 e 05 respectiviamente.

<div style="text-align: center;">
    <img src="/projetos/HTF/images/crowds_zara01.jpg" alt="Figura 04: Imagem do dataset UCY Zara 01" width="600"/>
    <p><em>Figura 04: Imagem do dataset UCY Zara 01</em></p>
</div>

<div style="text-align: center;">
    <img src="/projetos/HTF/images/students_003.jpg" alt="Figura 05:Imagem do dataset UCY Students 03" width="600"/>
    <p><em>Figura 05: Imagem do dataset UCY Students 03</em></p>
</div>



## Arquitetura do Modelo:



> A arquitetura definida da GAN é composta por três módulos principais: o módulo de embedding das interações espaço-temporais (STIM - Spatial-Temporal Interaction Embedding Module), o gerador e o discriminador.

![objetivo](/projetos/HTF/images/FIG02.png)

> Dentre as propriedades e características do modelo da Social GAN tem-se:
> 
> 1º - O modelo do gerador e discriminador em conjunto com a modelagem de interação espaço-temporal (STIM) são capazes de modelar as diversas interações da trajetória humana e seus acordos sociais implicitos;
>
> 2º - O mecanismo de amostragem colaborativa (CS) caracterizada pela dinâmica do gerador e discriminador garante que a distribuição sintética do gerador é próxima à distribuição real desconhecida, uma vez que o discriminador disciplina o aprendizado do gerador de forma que as amostras sintéticas futuras geradas sejam socialmente factíveis.
>
> 3º - Previne o colapso de modo - *mode collape* - caracterizado por amostras sintéticas geradas a partir de uma única distribuição.

> O gerador tem em sua constituição células LSTM em uma arquitetura de *variational auto-encoder* para codificar a sequência de embeddings de entrada dentro de um espaço latente e de dimensionalidade reduzida para geração de amostras sintéticas (z) garantindo que os dados sintéticos estão contidos dentro de um mesmo manifold. De forma similar, o discriminador da arquitetura de referência é constituído de células transformers para receber os embeddings reais e sintéticos gerados, cuja célula transformer proporciona uma característica elevada para modelar relações multimodais espaço-temporais para este tipo de problema, garantindo que a rede profunda possa adquirir propriedades avançadas da percepção do conjunto de dados da trajetória humana.



## Artigos de Referência:



| Artigo | Autor | Fonte | Descrição | Tipo |
|----------------------------------------------------------------------------------------------------------|-------------------------------------------|-----------------------------------|--------------------------------------------------------------------|----------------------------------------|
| **Social LSTM: Human Trajectory Prediction in Crowded Spaces** | Alexandre Alahi et al. | Stanford University | Propõe o uso de redes LSTM para prever trajetórias humanas. O artigo introduz o conceito de "Social Pooling" para capturar interações sociais. | *Artigo Seminal* |
| **Safety-Compliant Generative Adversarial Networks for Human Trajectory Forecasting** | Parth Kothari and Alexandre Alahi | EPFL | Tem por objetivo modelar interações espaciais e temporais entre pedestres usando LSTMs e Transformers, aborda limitações em técnicas anteriores. | *Artigo de Referência* |



## Ferramentas a serem utilizadas:



| **Ferramenta**               | **Descrição**                                                                                 | **Aplicação no Projeto**                                                                               |
|------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **Google Colab/Jupyter**     | Ambiente de desenvolvimento baseado em nuvem com suporte a GPUs                               | Treinamento de modelos e execução de experimentos com alta demanda computacional                       |
| **PyTorch**                  | Biblioteca de aprendizado profundo com suporte a redes neurais dinâmicas                      | Implementação e treinamento dos modelos LSTM e Transformers                                            |
| **TensorFlow**               | Biblioteca de aprendizado de máquina com suporte a redes neurais                              | Alternativa ao PyTorch para o desenvolvimento de modelos de previsão de trajetórias                    |
| **Keras**                    | API de alto nível para redes neurais, integrada ao TensorFlow                                 | Criação de redes LSTM para replicar o Social GAN                                                       |
| **NumPy**                    | Biblioteca fundamental para computação numérica com Python                                    | Manipulação de arrays de dados e operações matemáticas durante o pré-processamento                     |
| **Pandas**                   | Biblioteca para análise e manipulação de dados tabulares                                      | Manipulação dos datasets de trajetórias e organização de dados de entrada                              |
| **Seaborn**                  | Biblioteca de visualização baseada em Matplotlib                                              | Visualização de correlações e tendências nos dados                                                     |
| **Plotly**                   | Biblioteca para visualizações interativas                                                     | Criação de gráficos interativos para explorar visualmente as previsões e trajetórias                   |
| **NetworkX**                 | Biblioteca para modelagem e análise de redes gráficas                                         | Implementação de modelos baseados em grafos                                                            |



##  Proposta de Avaliação dos resultados:



Para a avaliação dos resultados, serão utilizadas diversas métricas que são relevantes na pesquisa de predição de trajetória humana e utilizadas em diversos artigos, sendo:

1 - **Top-K Trajectories:**  
O método Top-K gera várias trajetórias potenciais para o futuro, e a saída do modelo consiste nas K trajetórias mais prováveis, com base nos padrões e interações aprendidos. Essas previsões são então avaliadas com as seguintes métricas:

* 1a - **Erro de Deslocamento Médio (ADE - Average Displacement Error):**  
O ADE mede a distância média (L2 norm) entre todas as posições previstas e as trajetórias reais ao longo do tempo, com base nas K melhores previsões. Essa métrica fornece uma visão geral de quão próximas as trajetórias previstas estão das trajetórias reais dos pedestres. No entanto, o ADE não capta diretamente as interações entre pedestres, algo que modelos como o Social GAN busca melhorar por meio de técnicas como "Social Pooling" e redes de atenção espacial-temporal.

* 1b - **Erro de Deslocamento Final (FDE - Final Displacement Error):**  
O FDE, semelhante ao ADE, mede a distância entre a posição final das trajetórias previstas e a posição final real dos pedestres. Essa métrica é particularmente utilizada para avaliar a precisão do modelo ao prever a posição ao final do horizonte de previsão. Assim como o ADE, o FDE, isoladamente, não avalia as interações sociais entre pedestres.

2 - **Taxa de Colisão:**  
A Taxa de Colisão avalia a porcentagem de trajetórias previstas que resultam em colisões entre pedestres. Esta métrica é fundamental para verificar se o modelo gera trajetórias socialmente aceitáveis. Modelos que geram trajetórias com altas taxas de colisão indicam que o comportamento aprendido não está alinhado com interações sociais naturais, notáveis pela ausência de colisões em geral. No caso de modelos como o Social GAN, a consideração das interações espaciais e temporais é essencial para minimizar essas colisões.

3 - **Distância de Wasserstein:**  
A Distância de Wasserstein mede a diferença entre a distribuição de trajetórias reais (*ground truth*) e as trajetórias previstas (geradas sinteticamente) pelo modelo profundo, considerando suas distribuições ao longo do tempo. Esta métrica é particularmente importante para avaliar a qualidade da geração de dados sintéticos em modelos generativos implícitos, fornecendo uma medida de quão bem o modelo está replicando o comportamento real.



## Cronograma



|  Fase                                                                 | 03/09 | 10/09 | 17/09 | 24/09 | 01/10 | 08/10 | 15/10 | 22/10 | 29/10 | 05/11 | 12/11 | 19/11 | 26/11 |
|-----------------------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| **Fase 1: Estudo de Artigos e Apresentação da Proposta**              |   X   |   X   |   X   |       |       |       |       |       |       |       |       |       |       |
| **Fase 2: Implementação do Modelo e Estruturação dos Datasets**       |       |       |   X   |   X   |   X   |   X   |       |       |       |       |       |       |       |
| **Fase 2.1: Apresentação do Entregável 2**                            |       |       |       |       |       |   X   |       |       |       |       |       |       |       |
| **Fase 3: Desenvolvimento do Modelo e Avaliações**                    |       |       |       |       |       |   X   |   X   |   X   |   X   |   X   |   X   |   X   |       |
| **Fase 4: Entrega e Apresentação Final**                              |       |       |       |       |       |       |       |       |       |       |       |       |   X   |

> Datas importantes:
>
> * **10/09 (E1):** Apresentação da Proposta de Pesquisa
> * **08/10 (E2):** Apresentação do Entregável 2
> * **25/11 e 26/11 (E3):** Entrega e Apresentação Final



## Referências Bibliográficas



> [1] P. Kothari e A. Alahi, "Safety-Compliant Generative Adversarial Networks for Human Trajectory Forecasting," IEEE Transactions on Intelligent Transportation Systems, vol. PP, pp. 1-11, abr. 2023.
>  
> [2] A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, e A. Alahi, "Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
> 
> [3] Z. Lv, X. Huang, e W. Cao, "An improved GAN with transformers for pedestrian trajectory prediction models," International Journal of Intelligent Systems, vol. 37, nº 8, pp. 4417-4436, 2022.
> 
> [4] S. Eiffert, K. Li, M. Shan, S. Worrall, S. Sukkarieh and E. Nebot, "Probabilistic Crowd GAN: Multimodal Pedestrian Trajectory Prediction Using a Graph Vehicle-Pedestrian Attention Network," in IEEE Robotics and Automation Letters, vol. 5, no. 4, pp. 5026-5033, Oct. 2020.
>
> [5] A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, L. Fei-Fei, and S. Savarese, "Social LSTM: Human trajectory prediction in crowded spaces," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 961-971. [Online]. Available: https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf
