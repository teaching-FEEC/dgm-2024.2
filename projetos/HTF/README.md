# `<Redes Adversárias Generativas Sociais: Geração de Amostras Futuras para Predição da Trajetória Humana em Espaços Populados>`
# `<Redes Adversárias Generativas Sociais: Geração de Amostras Futuras para Predição da Trajetória Humana em Espaços Populados>`



## Apresentação



O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> |Nome  | RA | Especialização|
> |--|--|--|
> | Hiuri Santana de Noronha  | 229961  | Eng. Eletricista|
> | Thiago Belina Silva Ramos  | 203975  | Eng. Eletricista|



## Descrição Resumida do Projeto



> Com base em pesquisas recentes, o projeto de pesquisa proposto visa estudar e propor modelos profundos de redes Generativas Adversariais (GANs) para geração de amostras futuras para predição da trajetória humana em espaços populados. A arquitetura da GAN possui estruturas combinadas de modelos profundos aplicando-se desde o variational auto-encoder (VAE) e rede neural recorrente com células LSTM para que em conjunto possuam a capacidade de modelar as interações multimodais de espaço e tempo da trajetória de pedestres conforme os acordos sociais implícitos existentes em uma distribuição real desconhacida. O processo adversarial é capaz de gerar o que se denomina de amostragem colaborativa (CS) cujo discriminador (D) possui a tarefa primordial de disciplinar o gerador (G) e garantir que a distribuição de G é próxima à distribuição real desconhecida. Tal processo é capaz de gerar amostras confiáveis de trajetória humana em espaços populados e cujos modelos profundos devem ser avaliados apropriadamente de acordo com métricas bem estabelecidas na literatura. 

> Para o treinamento do modelo profundo será utilizado um conjunto de datasets e um benchmark para comparação da eficácia do modelo profundo. Datasets como o ETH (Zurich), UCY e o Stanford Drone estão entre os mais utilizados na literatura técnica e proporcionam um conjunto de vídeos com trajetórias reais de pedestres em meio à espaços populados. Ainda, há a possibilidade de empregar dataset simulado em ambiente computacional com alta definição como o Forking Paths que tem sido utilizado na pesquisa de modelos de predição de trajetória humana recente. Tais datasets proporcionam ao modelo profundo um conjunto de dados estruturados na forma de séries temporais contendo o posicionamento x e y de pedestres ao longo de uma via a um determinado tempo. Com isso, os modelos profundos podem ser treinados para modelar o conjunto de regras sociais implícitas existentes em trajetórias humanas.

> A figura abaixo mostra de forma simplificada e ilustrativa como se dá o processo de predição da trajetória humana em espaços populados, cujas linhas sólidas (azul, vermelha e verde) são a representação do caminho real percorrido pelo pedestre e as linhas tracejadas são a representação de amostras sintéticas multimodais (espaço-tempo) geradas a partir do modelo profundo livres de colisão.
>
>![objetivo](/projetos/HTF/E1/FIG01.png)
>
>Fonte: *Safety-Compliant Generative Adversarial Networks for Human Trajectory Forecasting(Parth Kothari and Alexandre Alahi, 2023)*

> A principal motivação para a realização deste trabalho esta na compreensão do que é uma nova e latente área para aplicações futuras em robótica social (robôs humanoides), cidades inteligentes e sistemas inteligentes de transporte.

> A seguir tem-se os links para o vídeo da apresentação do entregável I e para a apresentação em slides.
 
> [Vídeo da Apresentação](https://youtube.com.br/) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Link da Apresentação](https://docs.google.com/presentation/d/1vHEd9DeePXOXCwRnCimp_5Cx16KPEKW2/edit?usp=sharing&ouid=101073047652792710630&rtpof=true&sd=true)



## Metodologia Proposta



A metodologia proposta para esta pesquisa envolve as seguintes definições:

* Proposta de estudo e pesquisa do tópico de referência;
* Modelagem das interações espaço-temporais com os datasets de referência a partir da implementação de um módulo de interação espaço-temporal (STIM) a partir de redes recorrentes com células LSTM ou Transformers na estrutura do gerador e discriminador - para fins de captura de features da evolução espaço-temporal das interações sociais;
* Treinamento da rede GAN modificada, do gerador e discriminador, para predições multimodais a partir de amostras futuras geradas em conformidade com os acordos sociais implícitos;
* Análise quanti-quali para avaliação das métricas obtidas - amostras sintéticas geradas avaliadas sob métricas utilizadas em artigos de referência;
* Análise qualitativa para avaliação da factibilidade e realismo de resultados obtidos em etapa inicial de treinamento.



## Datasets:



> De forma introdutória, serão considerados os seguintes datasets:
>
> | Dataset | Fonte | Descrição |
> | --      |  --   |    --     |
> | [ETH Pedestrian](https://paperswithcode.com/dataset/eth) | *ETH*  |O conjunto de dados que contém 1.804 imagens em três videoclipes. Capturado de um equipamento estéreo montado no carro, com uma resolução de 640 x 480 (bayered) e uma taxa de quadros de 13-14 FPS. |
> | [UCY Crowds Data](https://graphics.cs.ucy.ac.cy/portfolio) | *UCY Graphics*  | Conjunto de dados em vídeo contendo de pessoas em movimento em meio a multidões.|
> | [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) | *Stanford*  | Conjunto de dados em larga escala que coleta imagens e vídeos de vários tipos de agentes que navegam em um ambiente externo do mundo real, como um campus universitário.|

> Dataset opcional
>
> | Dataset | Fonte | Descrição |
> | --      |  --   |    --     |
> | [Forking Paths](https://github.com/JunweiLiang/Multiverse) | *Carnegie Mellon University (CMU) & Google*  |Conjunto de dados que contém 1.804 imagens em três videoclipes. Capturado de um equipamento estéreo montado no carro, com uma resolução de 640 x 480 (bayered) e uma taxa de quadros de 13-14 FPS. |

> Benchmark
>
> | Benchmark | Fonte | Descrição |
> | --      |  --   |    --     |
> | [TrajNet++](https://github.com/vita-epfl/trajnetplusplusbaselines) | *EPFL*  | TrajNet++ é um benchmark de previsão de trajetória centrado em interação em larga escala que compreende cenários explícitos de agente-agente. |



## Arquitetura do Modelo:



> A arquitetura definida da GAN é composta por três módulos principais: o módulo de embedding das interações espaço-temporais (STIM - Spatial-Temporal Interaction Embedding Module), o gerador e o discriminador.

![objetivo](/projetos/HTF/E1/FIG02.png)

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
